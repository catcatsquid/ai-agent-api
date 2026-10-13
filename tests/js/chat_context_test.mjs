/**
 * 前端「对话上下文」持久化的行为测试（Node，无需浏览器）
 *
 * 为什么要有它：只把 thread_id 存进对话对象是不够的 —— 推理模式（plan / react）
 * 决定那个 thread 属于**哪张图**，后端两张图各有独立 checkpointer。
 * 模式不持久化的话，刷新后模式回到默认 plan，react 会话的 thread_id 在 plan 图里
 * 根本不存在，后端静默当成新会话：界面显示着历史，模型却一无所知。
 *
 * 这个测试直接读**真实页面**的内联脚本，把相关函数抽出来放进 vm，
 * 用桩 localStorage / DOM 跑通「写入 → 刷新 → 恢复」的往返，不靠人肉检查。
 *
 * 运行: node tests/js/chat_context_test.mjs
 */
import fs from 'fs';
import vm from 'vm';

let pass = 0, fail = 0;
function check(name, cond, extra = '') {
  if (cond) { pass++; console.log(`  [OK]   ${name}`); }
  else { fail++; console.log(`  [FAIL] ${name}  ${extra}`); }
}

const PAGE = 'course_agent/static/chat.html';
const html = fs.readFileSync(PAGE, 'utf8');

/** 按花括号配对切出一个完整函数 —— 别用"往后截 N 个字符"当边界 */
function extractFn(src, name) {
  const i = src.indexOf('function ' + name + '(');
  if (i === -1) throw new Error('页面上找不到函数 ' + name);
  const start = src.indexOf('{', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') {
      depth--;
      if (depth === 0) return src.slice(i, j + 1);
    }
  }
  throw new Error('函数 ' + name + ' 的花括号不配对');
}

const WANTED = ['readStoredReasoningMode', 'applyReasoningMode',
                'rememberThreadId', 'loadChat', 'createNewChat'];

function makeFakeBtn(mode) {
  const b = { mode, active: false };
  b.classList = { toggle: (cls, on) => { if (cls === 'active') b.active = !!on; } };
  b.getAttribute = (k) => (k === 'data-mode' ? mode : null);
  return b;
}

/**
 * 造一个沙箱并跑页面里抽出来的代码。
 * store 传进来的对象会被**直接使用**（不复制）—— 这样可以跨沙箱共用同一份
 * storage，用来模拟"同一个浏览器刷新页面"。
 */
function makeCtx(store = {}) {
  const buttons = [makeFakeBtn('plan'), makeFakeBtn('react')];
  const ctx = {
    localStorage: {
      getItem: (k) => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
    },
    document: {
      getElementById: () => ({ textContent: '', focus() {} }),
      querySelectorAll: () => buttons,
    },
    currentChatId: null,
    currentThreadId: null,
    currentTutorSessionId: null,
    currentAppMode: 'qa',
    chats: [],
    saveCount: 0,
    renderMessages: () => {},
    renderChatList: () => {},
    updateStatus: () => {},
  };
  ctx.getChat = (id) => ctx.chats.find((c) => c.id === id) || null;
  ctx.saveChats = () => { ctx.saveCount++; };
  vm.createContext(ctx);

  const code = [
    'var currentReasoningMode = readStoredReasoningMode();',   // 页面里的初始化那一行
    ...WANTED.map((n) => extractFn(html, n)),
    'globalThis.__api = {get mode() { return currentReasoningMode; },'
      + ' applyReasoningMode, rememberThreadId, loadChat, createNewChat};',
  ].join('\n');
  vm.runInContext(code, ctx);
  return { ctx, store, buttons, api: ctx.__api };
}

console.log(`\n前端对话上下文持久化（读真实页面：${PAGE}）\n`);

// ---- 1. 模式默认值 ----
{
  const { api } = makeCtx();
  check('没存过模式时默认 plan', api.mode === 'plan', `mode=${api.mode}`);
}

// ---- 2. 切换模式要持久化 + 按钮高亮 ----
{
  const { api, store, buttons } = makeCtx();
  api.applyReasoningMode('react');
  check('切换后内存里的模式变了', api.mode === 'react', `mode=${api.mode}`);
  check('切换后写进了 localStorage', store.reasoning_mode === 'react',
        `store=${JSON.stringify(store)}`);
  check('按钮高亮跟着切换',
        buttons.find((b) => b.mode === 'react').active
        && !buttons.find((b) => b.mode === 'plan').active,
        JSON.stringify(buttons.map((b) => [b.mode, b.active])));
}

// ---- 3. 真往返：写入 → 换一个新沙箱（=刷新页面，storage 共用）→ 能恢复 ----
{
  const store = {};
  const before = makeCtx(store);
  before.api.applyReasoningMode('react');       // 写入路径
  const after = makeCtx(store);                 // 刷新页面
  check('写进 storage 后，刷新页面能恢复出 react', after.api.mode === 'react',
        `mode=${after.api.mode} store=${JSON.stringify(store)}`);
}

// ---- 4. 收到 thread_id 时，要连同"哪个模式"一起记进对话对象 ----
{
  const { ctx, api } = makeCtx({ reasoning_mode: 'react' });
  api.applyReasoningMode('react');
  const chat = { id: 'c1', threadId: null, reasoningMode: null, messages: [] };
  ctx.chats.push(chat);
  ctx.currentChatId = 'c1';
  api.rememberThreadId('T-1');
  check('记住了 thread_id', ctx.currentThreadId === 'T-1' && chat.threadId === 'T-1',
        `current=${ctx.currentThreadId} chat=${chat.threadId}`);
  check('连同模式一起记住（thread 属于哪张图）', chat.reasoningMode === 'react',
        `chat.reasoningMode=${chat.reasoningMode}`);
  check('落盘了一次', ctx.saveCount >= 1, `saveCount=${ctx.saveCount}`);

  const before = ctx.saveCount;
  api.rememberThreadId('');
  check('空 thread_id 不写不存', ctx.saveCount === before && chat.threadId === 'T-1');
}

// ---- 5. 切回旧对话时，模式与 thread 一起恢复 ----
{
  const { ctx, api } = makeCtx({ reasoning_mode: 'plan' });
  const chat = { id: 'c1', title: '旧对话', threadId: 'T-9', reasoningMode: 'react',
                 messages: [{ role: 'user', content: 'hi' }] };
  ctx.chats.push(chat);
  ctx.currentChatId = null;
  api.loadChat('c1');
  check('loadChat 恢复了 thread_id', ctx.currentThreadId === 'T-9',
        `current=${ctx.currentThreadId}`);
  check('loadChat 也恢复了模式（否则那个 thread 后端不认）', api.mode === 'react',
        `mode=${api.mode}`);
}

// ---- 6. 新建对话要记下当时的模式 ----
{
  const { ctx, api } = makeCtx({ reasoning_mode: 'react' });
  const chat = api.createNewChat();
  check('新对话带上了当前模式', chat.reasoningMode === 'react',
        `chat.reasoningMode=${chat.reasoningMode}`);
  check('新对话的 thread 初始为空', chat.threadId === null);
  check('新对话进了列表', ctx.chats.length === 1);
}

console.log(`\n${'='.repeat(70)}`);
console.log(`结果: ${pass} 通过 / ${fail} 失败`);
console.log('='.repeat(70));
process.exit(fail === 0 ? 0 : 1);
