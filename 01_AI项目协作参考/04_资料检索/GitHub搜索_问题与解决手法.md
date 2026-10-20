# GitHub 调研手册 —— 搜索过程问题与解决手法

> **本文档可移植**:任何 Windows + Git Bash 环境下的 AI 助手都能直接复用;
> 其他操作系统按 §1 的降级链思路自行探测等价工具链。
> 来源:2026-09-29 用 GitHub API 调研"多 Agent 协作项目"的踩坑复盘。
> 目标:读完直接复用结论,不重新探测环境、不重复撞限流。更新:2026-09-29

---

## 0. 参考机环境事实(2026-09 实测;你的机器先照此探测一遍再用)

| 项 | 值 |
|---|---|
| Shell | Git Bash(msys,Windows 10/11) |
| **可用** | `perl` 5.38(JSON::PP 内置)、`curl`(mingw64)、`powershell.exe` |
| **不可用** | `python`(商店占位 stub,报 exit 49)、`node`(command not found)、`gh` CLI(未安装) |
| LWP::UserAgent | 模块存在,但发 HTTPS **静默返回空**(无报错、无数据)——**别用它抓 HTTPS** |
| 桌面真实路径 | 用 `powershell.exe -NoProfile -Command "[Environment]::GetFolderPath('Desktop')"` 确认,别猜(可能有 OneDrive 重定向,真实路径可能是 `C:\<用户名>\Desktop` 或 OneDrive 下) |

**工具降级链(实测结论)**:python ✗ → node ✗ → **perl ✓**;perl 内部再降级:
LWP 抓 HTTPS ✗ → **curl 抓取 ✓**,perl 只做 JSON 解析。

## 1. 核心手法 ★:curl 抓 + perl 解,抓解分离

不要试图用一个工具包办网络和解析。curl 负责网络,perl(JSON::PP)负责结构化:

```bash
# 解析脚本(存成 /tmp/ghparse.pl 可反复用)
cat > /tmp/ghparse.pl <<'EOF'
use strict; use warnings; use JSON::PP;
binmode STDOUT, ':utf8';          # 消除中文 Wide character 警告
local $/; my $j = decode_json(<STDIN>);
for my $it (@{$j->{items} || []}) {
    printf "%-52s %6d  created %s pushed %s [%s] %s\n",
        $it->{full_name}, $it->{stargazers_count},
        substr($it->{created_at},0,10), substr($it->{pushed_at},0,10),
        ($it->{language}//"-"), substr($it->{description}//"",0,115);
}
EOF

# 用法:curl 落盘或直接管道
curl -s -H "Accept: application/vnd.github+json" \
  "https://api.github.com/search/repositories?q=multi-agent+collaboration+created:%3E2026-04-01&sort=stars&order=desc&per_page=10" \
| perl /tmp/ghparse.pl
```

搜索查询限定词可自由叠加,一条查询顶多条:
`q=multi-agent+framework+created:%3E2026-01-01+stars:%3E200`、
`q=...+pushed:%3E2026-09-15`(确认"仍在活跃")。

## 2. 限流规则与应对 ★(本次最大坑)

| API | 未认证限额 | 超限表现 |
|---|---|---|
| search API(搜索) | **10 次/分钟** | 空结果 / 403 |
| core API(单仓库详情等) | **60 次/小时** | 返回 `{"message":"API rate limit exceeded for <IP>..."}`,且 HTTP 状态可能仍是 200 —— **必须检查返回体里有没有 message 字段** |

应对顺序:

1. **省着用**:多条搜索之间 `sleep 6-7`;限定词合并进单条查询;
   "每仓库一次"的查询(如 good-first-issue 计数)最烧配额,放最后、按需做。
2. **先看还剩多少、何时重置**:
   ```bash
   curl -s https://api.github.com/rate_limit | head -c 300   # 看 remaining / reset(epoch)
   date +%s                                                   # 与当前时间相减 = 还要等几秒
   ```
3. **core 60 次用完后别干等**(本次等了 25 分钟,不值):
   - `gh` CLI 自带认证可绕开(未认证限额的 60/小时是按 IP 算的)——参考机未装,你的环境装了优先用;
   - **降级用 WebFetch 抓 `github.com/<owner>/<repo>` HTML 页**:不走 API 限流,
     README 全文、星数、commit 活跃度、CONTRIBUTING 是否存在、架构说明全都能拿到,
     做深度调研时信息比 API 更全。API 留给"批量列表 + 排序"这类只有它能干的事。

## 3. WebFetch 并发限制

并行同时发多个 WebFetch 会撞 **"user concurrency limit exceeded"**(本次 4 个并发
挂了 2 个)→ **改为串行,一次一个**;每个 prompt 里把想问的问题一次问全,
减少往返次数。

## 4. 其他坑

- **perl 正则提取字段**:`perl -0777 -ne 'print $1 if /"open_issues_count":(\d+)/'`
  对大 JSON 可用,但字段顺序/格式一变就**静默失配**(本次全部输出空,排查半天才
  发现其实是限流返回体,不是正则错)。批量跑之前,先 `curl` 单个仓库看原始返回,
  确认结构再写正则;更要紧的是先确认不是限流假数据。
- **UTF-8 打印**:printf 输出中文描述会刷屏 `Wide character in printf` ——
  只是警告、数据无损;根治用 `binmode STDOUT, ':utf8';`(见 §1 脚本)。
- **验证"仍在进行中"**:API 的 `pushed_at` 字段适合批量过滤
  (`pushed:>YYYY-MM-DD`),单仓库确认活跃度/维护状态则看 HTML 页的 commits 数、
  open issues/PR、最近 release。

## 5. 本次实际工作流(复现用)

```
① 定查询词(3-4 组,覆盖:新项目 created:>、活跃项目 pushed:>、主题 topic:)
② curl+perl 批量搜索,取星数/创建/最近push/简介        ← search API,每条间隔 6-7s
③ 按星数+创建时间初筛,分档(起飞中 / 早期可贡献 / 老牌在维护)
④ 深挖候选:WebFetch 逐个抓仓库 HTML 页(串行!),核实
   贡献友好度(CONTRIBUTING/issue 数/维护者邀请)与架构亮点
⑤ 老牌框架用 WebSearch 交叉确认现状(如 AutoGen 已并入 microsoft/agent-framework)
⑥ 汇总成 .md 交付
```

> ④ 中若还需 API 补数据(如 per-repo issue 计数),先查 §2 的 rate_limit,
> 剩额不足就全部改走 HTML 页,不要边撞边等。
