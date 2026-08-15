#!/usr/bin/env python3
"""Eliminate SystemVerilog `return` statements (Yosys has no `return` keyword).

Yosys requires functions to assign to the function name (no `return`).  This
module rewrites `return`-using functions into a single-exit form:

    function automatic <T> <name>(...);
      <decls>
      logic __ret;
      __ret = 1'b0;
      <body: `return X;` -> `__ret = 1'b1;`, and every statement after the
       first return guarded with `if (!__ret) ...`>
    endfunction

Semantics preserved: the first `return` to execute wins; later statements are
skipped via the `__ret` flag.  Purely local, semantics-neutral desugaring.
"""
import re

IF_BEGIN_RE = re.compile(r'^(\s*)if\s*\(([^)]*)\)\s*begin\b(.*)$')
RETURN_RE = re.compile(r'\breturn\s*([^;]*);')


def _find_matching_end(lines):
    depth = 0
    for i, ln in enumerate(lines):
        opens = len(re.findall(r'\bbegin\b', ln))
        closes = len(re.findall(r'\bend\b', ln))
        depth += opens - closes
        if depth <= 0:
            idx = ln.rfind('end')
            return i, ln[idx + 3:].strip()
    return len(lines) - 1, ""


def _split_statements(lines):
    stmts, cur, depth = [], [], 0
    for ln in lines:
        cur.append(ln)
        code = ln.strip().split("//")[0].rstrip()
        depth += len(re.findall(r'\bbegin\b', code)) - len(re.findall(r'\bend\b', code))
        if depth <= 0 and (code.endswith(';') or code.endswith('end')):
            stmts.append(cur)
            cur = []
            depth = 0
    if cur:
        stmts.append(cur)
    # Merge a line starting with `else` into the preceding statement (the
    # two-line `if (...) begin ... end` + `else begin ... end` form).
    merged = []
    for st in stmts:
        if merged and st[0].strip().startswith("else"):
            merged[-1].extend(st)
        else:
            merged.append(st)
    return merged


def _guard(lines):
    lines[0] = re.sub(r'^(\s*)', r'\1if (!__ret) ', lines[0])
    return lines


def _replace_returns_inline(line, fnname):
    had = False

    def repl(m):
        nonlocal had
        had = True
        expr = m.group(1).strip()
        if expr == "" or expr == fnname:
            return "__ret = 1'b1;"
        return "begin %s = %s; __ret = 1'b1; end" % (fnname, expr)

    return RETURN_RE.sub(repl, line), had


def rewrite_statement(lines, fnname):
    """Rewrite a single statement (list of lines).  Returns (new_lines, had_return)."""
    if not any(re.search(r'\breturn\b', l) for l in lines):
        return lines, False

    m = IF_BEGIN_RE.match(lines[0])
    if m is None:
        out, had = [], False
        for ln in lines:
            ln2, h = _replace_returns_inline(ln, fnname)
            had = had or h
            out.append(ln2)
        return out, had

    indent, cond, inline = m.group(1), m.group(2), m.group(3)

    if inline.strip():
        body_text = inline
        end_idx = body_text.rfind('end')
        tail = body_text[end_idx + 3:].strip()
        body = body_text[:end_idx].strip()
        newbody, body_had = rewrite_statement([body], fnname)
        line = "%sif (%s) begin %s end" % (indent, cond, newbody[0].strip())
        if tail:
            line += " " + tail
        return [line], body_had

    end_i, tail = _find_matching_end(lines)
    body = lines[1:end_i]
    rest = lines[end_i + 1:]
    newbody, body_had = rewrite_block(body, fnname)
    out = [indent + "if (%s) begin" % cond]
    out.extend(newbody)
    endline = indent + "end"
    if tail:
        endline += " " + tail
    out.append(endline)
    out.extend(rest)
    return out, body_had


def rewrite_block(lines, fnname):
    stmts = _split_statements(lines)
    out = []
    past = False
    for st in stmts:
        if not st or all(not ln.strip() for ln in st):
            continue
        target = st
        if past:
            target = _guard(st)
        new, had = rewrite_statement(target, fnname)
        out.extend(new)
        if had:
            past = True
    return out, past


def eliminate_returns_in_body(body_text, fnname):
    lines = body_text.split("\n")
    out, _ = rewrite_block(lines, fnname)
    return "\n".join(out)


def eliminate_returns(text):
    def repl(m):
        block = m.group(0)
        if not re.search(r"\breturn\b", block):
            return block
        hdr_end = block.find(");")
        if hdr_end < 0:
            return block
        header = block[: hdr_end + 2]
        body = block[hdr_end + 2:]
        body = body[: body.rfind("endfunction")]
        head = header.replace("\n", " ")
        nm = re.search(r"(\w+)\s*\(", head)
        fnname = nm.group(1) if nm else "fn"
        indent = re.match(r"\s*", header.split("\n")[0]).group(0)
        newbody = eliminate_returns_in_body(body, fnname)
        decl = "%slogic __ret;" % indent
        init = "%s__ret = 1'b0;" % indent
        return (
            header + "\n" + decl + "\n" + init + "\n"
            + newbody.rstrip("\n") + "\n" + indent + "endfunction"
        )

    FUNCTION_RE = re.compile(r"[ \t]*function\b.*?endfunction", re.DOTALL)
    return FUNCTION_RE.sub(repl, text)
