"""PilKB 真实 LLM 实跑脚本: 面向已有 kb 连续发问, 检验检索与生成的端到端效果。

前提: 服务以 PILKB_FAKE_LLM=0 启动, 且 kb-demo 已完成 面试演示产品方案.docx 的索引。
"""
import json
import time
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"
KB_ID = "kb-demo"


def _post(url, payload, token):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def _get(url, token):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, resp.read().decode("utf-8", "replace")


def main():
    st, body = _post(f"{BASE}/api/v1/auth/dev-token", None, None)
    token = json.loads(body)["access_token"]
    print("STEP: 当前索引状态 ->", _get(f"{BASE}/api/v1/documents/list?kb_id={KB_ID}", token)[1])

    questions = [
        "这份产品方案的核心定位与目标用户群体是什么？",
        "企业版年费多少？年度促销如何计费？客户购买 20 个账号合计多少钱？",
        "PilKB 的混合检索是如何实现的？重排序环节用的什么模型？",
    ]
    for i, q in enumerate(questions, 1):
        print("\n" + "=" * 72)
        print(f"Q{i}: {q}")
        print("=" * 72)
        t0 = time.time()
        st, body = _post(
            f"{BASE}/api/v1/chat",
            {"question": q, "kb_id": KB_ID, "top_k": 3, "mode": "rag"},
            token,
        )
        print(f"[HTTP {st} | {time.time()-t0:.1f}s]")
        if st != 200:
            print("ERR:", body); continue
        data = json.loads(body)
        print("A: " + data.get("answer", "").strip())
        print(f"\n引用条目 (共 {len(data.get('sources', []))} 条):")
        for j, s in enumerate(data.get("sources", []), 1):
            print(f"  [{j}] score={s.get('score',0):.3f} | {s.get('content','')[:160].replace(chr(10),' ')}")


if __name__ == "__main__":
    main()
