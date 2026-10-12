/**
 * 前端认证块的行为测试（Node，无需浏览器）
 *
 * 为什么要有它：第一版登录框用了 window.prompt()，在被禁用原生弹窗的环境里
 * 直接抛 "prompt() is not supported"，页面完全发不出消息。这个测试直接读取
 * **两个页面的真实内联脚本**（不是副本，避免测试与产物脱节），用最小 DOM shim
 * 跑通登录链路，并硬性禁止 prompt/alert/confirm。
 *
 * 运行: node tests/js/auth_block_test.mjs
 */
import fs from 'fs';
import vm from 'vm';
import path from 'path';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  [OK]   ${name}`); }
  else { fail++; console.log(`  [FAIL] ${name}  ${extra}`); }
}
const tick = () => new Promise((r) => setTimeout(r, 0));

const PAGES = [
  'course_agent/static/chat.html',
  'course_agent/static/kb.html',
];

/** 从页面里切出认证块（从注释横幅到 let authToken = ... 为止） */
function extractAuthBlock(html) {
  const marker = html.indexOf('// 认证：JWT 登录');
  if (marker === -1) return null;
  const start = html.lastIndexOf('// ====', marker);
  const endMarker = 'let authToken = authHeaders();';
  const end = html.indexOf(endMarker);
  if (start === -1 || end === -1) return null;
  return html.slice(start, end + endMarker.length);
}

/** 去掉注释后再做"禁用 API"检查 —— 否则解释"为什么不能用 prompt()"的注释会误报 */
function stripComments(code) {
  return code
    .replace(/\/\*[\s\S]*?\*\//g, ' ')      // 块注释
    .replace(/(^|[^:'"\\])\/\/[^\n]*/g, '$1'); // 行注释（避开 'https://' 里的 //）
}

function makeEl(tag) {
  const el = {
    tagName: tag, children: [], style: {}, _text: '', parent: null,
    value: '', type: '', placeholder: '',
    onclick: null, onkeydown: null, removed: false, focused: false,
    set textContent(v) { this._text = v; }, get textContent() { return this._text; },
    appendChild(c) { c.parent = this; this.children.push(c); return c; },
    remove() {
      this.removed = true;
      if (this.parent) {
        const i = this.parent.children.indexOf(this);
        if (i >= 0) this.parent.children.splice(i, 1);
      }
    },
    focus() { this.focused = true; },
  };
  return el;
}
const mkStorage = () => {
  const m = new Map();
  return {
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
};

function makeSandbox(block, { loginSucceeds = true } = {}) {
  const body = makeEl('body');
  const calls = [];
  const nativeFetch = async (url, init) => {
    const u = typeof url === 'string' ? url : url.url;
    const headers = (init && init.headers) || {};
    calls.push({ url: u, headers });
    if (u === '/auth/login') {
      return {
        status: 200,
        json: async () => (loginSucceeds
          ? { code: 0, msg: '登录成功', token: 'TEST-TOKEN' }
          : { code: 401, msg: '用户名或密码错误' }),
      };
    }
    const authed = headers['Authorization'] === 'Bearer TEST-TOKEN';
    return { status: authed ? 200 : 401, json: async () => ({ code: authed ? 0 : 401 }) };
  };
  const sandbox = {
    document: { createElement: makeEl, body, getElementById: () => null },
    console, setTimeout, clearTimeout, Promise, Object, Error, JSON, String, Boolean,
    sessionStorage: mkStorage(), localStorage: mkStorage(),
  };
  sandbox.window = sandbox;
  sandbox.window.fetch = nativeFetch;
  vm.createContext(sandbox);
  vm.runInContext(block, sandbox);
  const dialogParts = () => {
    const overlay = body.children[body.children.length - 1];
    const box = overlay.children[0];
    return {
      overlay, box,
      input: box.children[2], err: box.children[3],
      cancel: box.children[4].children[0], submit: box.children[4].children[1],
    };
  };
  return { sandbox, body, calls, dialogParts };
}

const blocks = {};
for (const page of PAGES) {
  if (!fs.existsSync(page)) {
    console.log(`\n[跳过] 找不到 ${page}（请从项目根目录运行）`);
    process.exit(1);
  }
  const html = fs.readFileSync(page, 'utf8');
  const block = extractAuthBlock(html);
  console.log(`\n${'='.repeat(70)}\n${page}  (认证块 ${block ? block.length : 0} 字符)\n${'='.repeat(70)}`);
  if (!block) {
    check('能从页面里切出认证块', false, '标记没找到');
    continue;
  }
  blocks[page] = block;

  console.log('1) 绝不使用原生弹窗（在不支持的环境里会直接抛错）');
  const codeOnly = stripComments(block);
  check('没有 window.prompt(', !codeOnly.includes('window.prompt('));
  check('没有 prompt(', !/(^|[^.\w])prompt\s*\(/.test(codeOnly));
  check('没有 alert(', !/(^|[^.\w])alert\s*\(/.test(codeOnly));
  check('没有 confirm(', !/(^|[^.\w])confirm\s*\(/.test(codeOnly));
  check('提供了自绘 askPassword', typeof makeSandbox(block).sandbox.askPassword === 'function');

  console.log('\n2) 输入密码 → 点登录 → 返回密码并关闭对话框');
  {
    const { sandbox, body, dialogParts } = makeSandbox(block);
    const p = sandbox.askPassword();
    await tick();
    const { input, submit } = dialogParts();
    check('对话框已插入页面', body.children.length === 1);
    check('输入框自动聚焦', input.focused === true);
    check('密码框类型为 password', input.type === 'password');
    input.value = 'secret';
    submit.onclick();
    check('返回输入的密码', (await p) === 'secret');
    check('对话框已从 DOM 移除', body.children.length === 0);
  }

  console.log('\n3) 取消 / 点遮罩 → null 且关闭');
  {
    const A = makeSandbox(block);
    const p1 = A.sandbox.askPassword();
    await tick();
    A.dialogParts().cancel.onclick();
    check('取消返回 null', (await p1) === null);
    check('取消后关闭', A.body.children.length === 0);

    const B = makeSandbox(block);
    const p2 = B.sandbox.askPassword();
    await tick();
    B.dialogParts().overlay.onclick({ target: B.dialogParts().overlay });
    check('点遮罩返回 null', (await p2) === null);
  }

  console.log('\n4) 空密码不允许提交');
  {
    const { sandbox, body, dialogParts } = makeSandbox(block);
    const p = sandbox.askPassword();
    await tick();
    const { input, submit, err } = dialogParts();
    submit.onclick();
    check('给出提示且不关闭', err.textContent.includes('请输入密码') && body.children.length === 1,
      `err='${err.textContent}'`);
    input.value = 'x';
    submit.onclick();
    check('填了才放行', (await p) === 'x');
  }

  console.log('\n5) 401 → 自动登录 → 带令牌重试');
  {
    const { sandbox, body, calls, dialogParts } = makeSandbox(block);
    const req = sandbox.window.fetch('/api/thing');
    await tick(); await tick();
    check('已弹出登录框', body.children.length === 1);
    const { input, submit } = dialogParts();
    input.value = 'secret';
    submit.onclick();
    const res = await req;
    check('重试后 200', res.status === 200, `status=${res.status}`);
    check('顺序：原接口 → login → 重试',
      JSON.stringify(calls.map((c) => c.url)) ===
      JSON.stringify(['/api/thing', '/auth/login', '/api/thing']),
      JSON.stringify(calls.map((c) => c.url)));
    check('重试带上 Authorization',
      calls[2].headers['Authorization'] === 'Bearer TEST-TOKEN');
  }

  console.log('\n6) 登录失败 / 取消 → 可读错误，不抛裸 401');
  {
    const A = makeSandbox(block, { loginSucceeds: false });
    const reqA = A.sandbox.window.fetch('/api/thing');
    await tick(); await tick();
    const pa = A.dialogParts();
    pa.input.value = 'wrong';
    pa.submit.onclick();
    let msgA = '';
    try { await reqA; } catch (e) { msgA = e.message; }
    check('登录失败给出可读错误', msgA.includes('未登录'), `msg=${msgA}`);

    const B = makeSandbox(block);
    const reqB = B.sandbox.window.fetch('/api/thing');
    await tick(); await tick();
    B.dialogParts().cancel.onclick();
    let msgB = '';
    try { await reqB; } catch (e) { msgB = e.message; }
    check('取消登录给出可读错误', msgB.includes('未登录'), `msg=${msgB}`);
  }

  console.log('\n7) 令牌注入与存储降级');
  {
    const { sandbox } = makeSandbox(block);
    check('初始不带 Authorization', !sandbox.authHeaders()['Authorization']);
    sandbox.authTokenSet('ABC');
    check('带 Bearer 令牌', sandbox.authHeaders()['Authorization'] === 'Bearer ABC');
    sandbox.localStorage.setItem('api_key', 'K');
    check('兼容 X-API-Key', sandbox.authHeaders()['X-API-Key'] === 'K');

    sandbox.sessionStorage = {
      getItem() { throw new Error('SecurityError'); },
      setItem() { throw new Error('SecurityError'); },
    };
    let threw = false;
    try { sandbox.authTokenSet('MEM'); } catch (e) { threw = true; }
    check('sessionStorage 不可用时不抛异常', threw === false);
    check('仍能读回令牌（内存兜底）',
      sandbox.authHeaders()['Authorization'] === 'Bearer MEM');
  }
}

console.log(`\n${'='.repeat(70)}\n结果: ${pass} 通过 / ${fail} 失败\n${'='.repeat(70)}`);
process.exit(fail ? 1 : 0);
