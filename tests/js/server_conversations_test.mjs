/**
 * 前端「服务端会话导入」的行为测试（Node，无需浏览器）
 *
 * 为什么要有它：本地 chats 只活在 localStorage 里（换设备、清缓存就没了）。
 * 1.2.0 起后端把会话状态落盘了，前端要把"服务端有、本地没有"的会话导进来。
 * 导入必须带上 **threadId 与 reasoningMode** —— 少了任何一个，点进去都恢复不了
 * 那条会话（模式错了 thread 就落在另一张图上，后端 404 或静默新开）。
 *
 * 这个测试读**真实页面**，把函数抽出来放进 vm，用桩 fetch / localStorage 跑，
 * 覆盖：正常导入、已存在不重复导、删过的不再导、网络失败静默放弃、
 * 单条失败不影响其它、没有可展示消息就跳过、流式过程中不打扰。
 *
 * 运行: node tests/js/server_conversations_test.mjs
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
  const sig = 'function ' + name + '(';
  const i = src.indexOf(sig);
  if (i === -1) throw new Error('页面上找不到函数 ' + name);
  // ⚠️ 别把 async 丢了：被抽的函数里有 await，而 await 只允许出现在 async 函数里，
  // 抽成普通函数会直接 SyntaxError（"await is only valid in async functions"）。
  const isAsync = /\basync\s*$/.test(src.slice(Math.max(0, i - 10), i));
  const start = src.indexOf('{', i);
  let depth = 0;
  for (let j = start; j < src.length; j++) {
    if (src[j] === '{') depth++;
    else if (src[j] === '}') {
      depth--;
      if (depth === 0) return (isAsync ? 'async ' : '') + src.slice(i, j + 1);
    }
  }
  throw new Error('函数 ' + name + ' 的花括号不配对');
}

const LIST_URL = '/chat/conversations?limit=20';

function listOf(...rows) {
  return { code: 0, data: rows };
}
function detailOf(thread_id, mode, messages) {
  return { code: 0, data: { thread_id, mode, messages } };
}

/**
 * 造沙箱：responses 里的 key 是**完整 URL**，值是返回的 JSON；
 * 传 'throw' 表示这个请求抛异常；缺失的 key 视为 404。
 * store 传进来的对象会被直接使用 —— 可以跨沙箱共用，模拟"同一个浏览器刷新"。
 */
function makeCtx({ responses = {}, chats = [], dismissed = [], isStreaming = false,
                   store = {} } = {}) {
  const calls = [];
  const opts = [];
  const ctx = {
    localStorage: {
      getItem: (k) => (k in store ? store[k] : null),
      setItem: (k, v) => { store[k] = String(v); },
    },
    document: { getElementById: () => ({ textContent: '', focus() {} }) },
    fetch: async (url, init) => {
      calls.push(url);
      opts.push(init || null);
      const hit = responses[url];
      if (hit === undefined) return { ok: false, status: 404, json: async () => ({}) };
      if (hit === 'throw') throw new Error('network down');
      return { ok: true, status: 200, json: async () => hit };
    },
    chats,
    dismissedThreads: dismissed,
    isStreaming,
    currentChatId: null,
    currentThreadId: null,
    currentTutorSessionId: null,
    currentAppMode: 'qa',
    saveCount: 0,
    renderCount: 0,
    toastCount: 0,
    saveChats() { ctx.saveCount++; },
    renderChatList() { ctx.renderCount++; },
    renderMessages() {},
    updateStatus() {},
    showToast() { ctx.toastCount++; },
  };
  vm.createContext(ctx);
  vm.runInContext(
    ['importServerConversations', 'deleteServerConversation', 'forgetDismissed',
     'deleteChat']
      .map((n) => extractFn(html, n)).join('\n')
    + '\nglobalThis.__import = importServerConversations;'
    + '\nglobalThis.__deleteChat = deleteChat;'
    + '\nglobalThis.__deleteServer = deleteServerConversation;',
    ctx);
  return { ctx, calls, opts, store };
}

const PLAN_CONV = { thread_id: 'T-1', mode: 'plan', title: '什么是死锁？',
                    updated_at: '2026-10-13T04:00:00+00:00' };
const PLAN_DETAIL = detailOf('T-1', 'plan', [
  { role: 'user', content: '什么是死锁？' },
  { role: 'assistant', content: '死锁是……' },
]);

console.log(`\n前端服务端会话导入（读真实页面：${PAGE}）\n`);

// ---- 1. 正常导入：threadId 与模式都必须带上 ----
{
  const { ctx } = makeCtx({
    responses: {
      [LIST_URL]: listOf(PLAN_CONV),
      '/chat/conversations/T-1?mode=plan': PLAN_DETAIL,
    },
  });
  await ctx.__import();
  check('导入了 1 条', ctx.chats.length === 1, `chats=${ctx.chats.length}`);
  const c = ctx.chats[0] || {};
  check('带上了 thread_id（否则点进去是新会话）', c.threadId === 'T-1',
        `threadId=${c.threadId}`);
  check('带上了推理模式（thread 属于哪张图）', c.reasoningMode === 'plan',
        `reasoningMode=${c.reasoningMode}`);
  check('消息来自服务端且顺序正确',
        c.messages && c.messages.length === 2 && c.messages[0].content === '什么是死锁？',
        JSON.stringify(c.messages));
  check('标题用了服务端给的', c.title === '什么是死锁？', `title=${c.title}`);
  check('打了"来自服务端"的标记', c.fromServer === true);
  check('导入后落盘了一次', ctx.saveCount === 1, `saveCount=${ctx.saveCount}`);
  check('导入后重渲染了列表', ctx.renderCount >= 1, `renderCount=${ctx.renderCount}`);
}

// ---- 2. 本地已有同一个 thread 时不重复导入 ----
{
  const local = [{ id: 'x', threadId: 'T-1', title: '本地已有的标题', messages: [] }];
  const { ctx, calls } = makeCtx({
    chats: local,
    responses: {
      [LIST_URL]: listOf(PLAN_CONV),
      '/chat/conversations/T-1?mode=plan': PLAN_DETAIL,
    },
  });
  await ctx.__import();
  check('没有重复导入', ctx.chats.length === 1, `chats=${ctx.chats.length}`);
  check('本地那条没被覆盖', ctx.chats[0].title === '本地已有的标题');
  check('连详情都没去请求（省一次往返）',
        !calls.includes('/chat/conversations/T-1?mode=plan'), JSON.stringify(calls));
}

// ---- 3. 用户删过的会话不再导回来 ----
{
  const { ctx } = makeCtx({
    dismissed: ['T-1'],
    responses: {
      [LIST_URL]: listOf(PLAN_CONV),
      '/chat/conversations/T-1?mode=plan': PLAN_DETAIL,
    },
  });
  await ctx.__import();
  check('删过的会话没有被导回来', ctx.chats.length === 0, `chats=${ctx.chats.length}`);
}

// ---- 4. 网络失败要静默放弃，不能把页面搞崩 ----
{
  const { ctx } = makeCtx({ responses: { [LIST_URL]: 'throw' } });
  let threw = null;
  try { await ctx.__import(); } catch (e) { threw = e; }
  check('列表请求抛异常时不往外冒', threw === null, `threw=${threw}`);
  check('也没有改动本地列表', ctx.chats.length === 0);
}

// ---- 5. 单条详情失败，不影响其它会话 ----
{
  const good = { thread_id: 'T-2', mode: 'react', title: '第二个',
                 updated_at: '2026-10-13T05:00:00+00:00' };
  const { ctx } = makeCtx({
    responses: {
      [LIST_URL]: listOf(PLAN_CONV, good),
      // T-1 的详情故意缺失（→ 404）；T-2 正常
      '/chat/conversations/T-2?mode=react':
        detailOf('T-2', 'react', [{ role: 'user', content: '第二个' }]),
    },
  });
  await ctx.__import();
  check('取不到详情的那条被跳过，另一条照样导入',
        ctx.chats.length === 1 && ctx.chats[0].threadId === 'T-2',
        JSON.stringify(ctx.chats.map((c) => c.threadId)));
  check('react 的模式被记下来了', ctx.chats[0].reasoningMode === 'react',
        `reasoningMode=${ctx.chats[0].reasoningMode}`);
}

// ---- 6. 没有可展示消息的会话不导入（避免导入出空对话）----
{
  const { ctx } = makeCtx({
    responses: {
      [LIST_URL]: listOf(PLAN_CONV),
      '/chat/conversations/T-1?mode=plan': detailOf('T-1', 'plan', []),
    },
  });
  await ctx.__import();
  check('空消息的会话不会被导入', ctx.chats.length === 0, `chats=${ctx.chats.length}`);
}

// ---- 7. 正在流式回答时不要打扰 ----
{
  const { ctx, calls } = makeCtx({
    isStreaming: true,
    responses: { [LIST_URL]: listOf(PLAN_CONV) },
  });
  await ctx.__import();
  check('流式过程中一个请求都不发', calls.length === 0, JSON.stringify(calls));
}

// ---- 8. 删除会话要同步到服务端 ----
{
  const store = {};
  const { ctx, calls, opts } = makeCtx({
    store,
    responses: { '/chat/conversations/T-9?mode=react': { code: 0, data: { deleted: true } } },
  });
  ctx.chats.push({ id: 'c1', threadId: 'T-9', reasoningMode: 'react',
                   title: '要删掉的', messages: [] });
  ctx.currentChatId = null;
  ctx.__deleteChat('c1');

  check('本地删掉了', ctx.chats.length === 0, `chats=${ctx.chats.length}`);
  check('发出了 DELETE 请求（本地删 ≠ 服务端删）',
        opts.some((o) => o && o.method === 'DELETE'),
        JSON.stringify(opts));
  check('请求带上了正确的 mode（两张图各有各的库）',
        calls.some((u) => u.includes('/chat/conversations/T-9') && u.includes('mode=react')),
        JSON.stringify(calls));
  check('记进了 dismissed（服务端删失败也不会被导回来）',
        ctx.dismissedThreads.includes('T-9'), JSON.stringify(ctx.dismissedThreads));
  check('dismissed 落了盘', !!store.dismissed_threads, JSON.stringify(store));
}

// ---- 9. 服务端删失败（网络断）不能影响本地删除 ----
{
  const { ctx } = makeCtx({
    responses: { '/chat/conversations/T-8?mode=plan': 'throw' },
  });
  ctx.chats.push({ id: 'c2', threadId: 'T-8', reasoningMode: 'plan',
                   title: 'y', messages: [] });
  ctx.currentChatId = null;
  let threw = null;
  try { ctx.__deleteChat('c2'); } catch (e) { threw = e; }
  await new Promise((r) => setTimeout(r, 0));     // 让 fire-and-forget 的 promise 跑完
  check('网络失败时本地照样删掉', ctx.chats.length === 0, `chats=${ctx.chats.length}`);
  check('没有把异常抛给调用方', threw === null, `threw=${threw}`);
  check('仍然记进 dismissed（下次加载不会再导回来）',
        ctx.dismissedThreads.includes('T-8'), JSON.stringify(ctx.dismissedThreads));
}

// ---- 10. 还没聊过的本地对话不发删除请求 ----
{
  const { ctx, calls } = makeCtx();
  ctx.chats.push({ id: 'c3', threadId: null, title: '还没聊过', messages: [] });
  ctx.currentChatId = null;
  ctx.__deleteChat('c3');
  check('没有 thread 的对话不发删除请求', calls.length === 0, JSON.stringify(calls));
  check('本地照删', ctx.chats.length === 0);
}

// ---- 11. deleteServerConversation 本身 ----
{
  const { ctx, calls } = makeCtx();     // 未定义的 URL → 桩返回 404
  check('服务端本来就没有（404）也算"已经干净了"',
        (await ctx.__deleteServer('T-X', 'plan')) === true);
  const before = calls.length;
  check('没有 thread 时直接返回 true', (await ctx.__deleteServer(null, 'plan')) === true);
  check('没有 thread 时不发请求', calls.length === before, JSON.stringify(calls));
}

// ---- 12. 自我清洁：服务端确认删掉后，黑名单记录就该被收回去 ----
{
  const store = {};
  const { ctx } = makeCtx({
    store,
    responses: { '/chat/conversations/T-7?mode=plan': { code: 0, data: { deleted: true } } },
  });
  ctx.chats.push({ id: 'c7', threadId: 'T-7', reasoningMode: 'plan',
                   title: 'z', messages: [] });
  ctx.currentChatId = null;
  ctx.__deleteChat('c7');

  check('删除的瞬间**先**记进黑名单（服务端万一步失败，这是保险）',
        ctx.dismissedThreads.includes('T-7'), JSON.stringify(ctx.dismissedThreads));
  await new Promise((r) => setTimeout(r, 0));      // 等服务端确认
  check('服务端确认删掉后，黑名单自我清洁掉（那个 thread 已不可能再出现）',
        !ctx.dismissedThreads.includes('T-7'), JSON.stringify(ctx.dismissedThreads));
  check('清洁后的状态也落了盘',
        !JSON.parse(store.dismissed_threads || '[]').includes('T-7'),
        store.dismissed_threads);
}

// ---- 13. 反例：服务端**没**删掉时，黑名单必须留着 ----
{
  const { ctx } = makeCtx({
    responses: { '/chat/conversations/T-6?mode=plan': 'throw' },
  });
  ctx.chats.push({ id: 'c6', threadId: 'T-6', reasoningMode: 'plan',
                   title: 'w', messages: [] });
  ctx.currentChatId = null;
  ctx.__deleteChat('c6');
  await new Promise((r) => setTimeout(r, 0));
  check('服务端删失败时黑名单必须留着（否则下次加载又导回来）',
        ctx.dismissedThreads.includes('T-6'), JSON.stringify(ctx.dismissedThreads));
}

console.log(`\n${'='.repeat(70)}`);
console.log(`结果: ${pass} 通过 / ${fail} 失败`);
console.log('='.repeat(70));
process.exit(fail === 0 ? 0 : 1);
