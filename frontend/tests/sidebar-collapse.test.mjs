import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import React from 'react';
import ReactDOMServer from 'react-dom/server';
import ts from 'typescript';

const require = createRequire(import.meta.url);
const key = 'ai-tutor-sidebar-collapsed';
const source = readFileSync(new URL('../src/components/CollapsibleSidebar.tsx', import.meta.url), 'utf8');

function harness() {
  let state = false;
  let mounted = false;
  let effect;
  const react = { ...React,
    useState: initial => [mounted ? state : (state = initial), value => { state = value; }],
    useEffect: callback => { if (!mounted) effect = callback; },
  };
  const code = ts.transpileModule(source, { compilerOptions: {
    module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React,
    target: ts.ScriptTarget.ES2020, esModuleInterop: true,
  } }).outputText;
  const module = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => name === 'react' ? react : require(name), module, module.exports);
  const children = React.createElement('aside', null, 'Workspaces, Threads, Docs, New Chat, History');
  return {
    render: () => module.exports.CollapsibleSidebar({ children }),
    mount: () => { mounted = true; effect(); },
    children,
  };
}

function parts(tree) {
  const [button, navigation] = React.Children.toArray(tree.props.children);
  return { button, navigation };
}

test('SSR shows sidebar by default without accessing window', () => {
  const previous = globalThis.window;
  delete globalThis.window;
  try {
    const tree = harness().render();
    assert.equal(parts(tree).button.props['aria-expanded'], true);
    assert.equal(parts(tree).navigation.props.hidden, false);
    assert.match(ReactDOMServer.renderToStaticMarkup(tree), /Workspaces, Threads, Docs/);
  } finally { globalThis.window = previous; }
});

test('collapse frees all sidebar width, retains toggle and children, then restores navigation', () => {
  const previous = globalThis.window;
  const storage = new Map();
  globalThis.window = { localStorage: { getItem: name => storage.get(name), setItem: (name, value) => storage.set(name, value) } };
  try {
    const component = harness();
    component.render();
    component.mount();
    let tree = component.render();
    assert.match(tree.props.className, /w-64/);
    let { button, navigation } = parts(tree);
    assert.equal(button.props.type, 'button'); // Native Enter/Space activation.
    assert.equal(button.props['aria-controls'], navigation.props.id);
    assert.equal(button.props['aria-label'], 'Collapse sidebar');
    button.props.onClick();
    tree = component.render();
    ({ button, navigation } = parts(tree));
    assert.match(tree.props.className, /\bw-0\b/);
    assert.doesNotMatch(tree.props.className, /w-64/);
    assert.equal(navigation.props.hidden, true);
    assert.equal(navigation.props.children, component.children);
    assert.equal(button.props['aria-expanded'], false);
    assert.equal(button.props['aria-label'], 'Expand sidebar');
    assert.equal(storage.get(key), 'true');
    assert.match(ReactDOMServer.renderToStaticMarkup(tree), /aria-label="Expand sidebar"/);
    button.props.onClick();
    tree = component.render();
    assert.match(tree.props.className, /w-64/);
    assert.equal(parts(tree).navigation.props.hidden, false);
    assert.equal(parts(tree).button.props['aria-expanded'], true);
    assert.equal(storage.get(key), 'false');
  } finally { globalThis.window = previous; }
});

test('reload restores preference only after hydration; stale values stay expanded', () => {
  const previous = globalThis.window;
  try {
    for (const saved of ['true', 'false', null, 'invalid']) {
      globalThis.window = { localStorage: { getItem: () => saved } };
      const component = harness();
      assert.equal(parts(component.render()).navigation.props.hidden, false);
      component.mount();
      assert.equal(parts(component.render()).navigation.props.hidden, saved === 'true');
    }
  } finally { globalThis.window = previous; }
});

test('blocked browser storage does not prevent either toggle direction', () => {
  const previous = globalThis.window;
  globalThis.window = { localStorage: { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } } };
  try {
    const component = harness();
    component.render();
    component.mount();
    parts(component.render()).button.props.onClick();
    assert.equal(parts(component.render()).navigation.props.hidden, true);
    parts(component.render()).button.props.onClick();
    assert.equal(parts(component.render()).navigation.props.hidden, false);
  } finally { globalThis.window = previous; }
});

test('small screens keep the restore control; split view retains mobile tabs and resize logic', () => {
  assert.doesNotMatch(source, /(?:sm|md|lg):hidden/);
  assert.match(source, /max-w-\[80vw\]/);
  const split = readFileSync(new URL('../src/components/WorkspaceSplitView.tsx', import.meta.url), 'utf8');
  assert.match(split, /flex-1 min-w-0/);
  assert.match(split, /lg:hidden flex items-center/);
  assert.match(split, /setMobileTab\("viewer"\)/);
  assert.match(split, /setMobileTab\("chat"\)/);
  assert.match(split, /getBoundingClientRect\(\)/);
  assert.match(split, /setSplitRatio\(newWidthPct\)/);
});
