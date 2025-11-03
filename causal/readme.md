# Causal Benchmark Quick Run

在仓库根目录运行以下命令。

## 运行因果图基准（Reasoned Strict + 并发）

```
python causal/run_causal_benchmark.py --config causal/config/config_deepseek.yaml --dataset datasets/causal_hybrid_n4_obs3_fixed.json --n-samples 30 --query-multiplier 1.0 --concurrency 8 --max-retries 1 --seed 42 --prompt-mode reasoned-strict --system-prompt-mode enhanced
```

- 关键参数：
  - `--concurrency 8` 并发请求加速批量推理。
  - `--seed 42` 保持可复现。
  - `--prompt-mode reasoned-strict`、`--system-prompt-mode enhanced` 为严格推理与增强系统提示配置。

## 生成比较图（六个配置）

```
python causal/plot_benchmark_comparison.py --inputs results/causal_hybrid_n4_obs3_fixed_deepseek-chat_baseline_q1p0x_seed42_temp0p7_c8_r1.json results/causal_hybrid_n4_obs3_fixed_deepseek-chat_baseline_sysbasic_q1p0x_seed42_temp1p5_c8_r1.json results/causal_hybrid_n4_obs3_fixed_deepseek-chat_baseline_sysenhanced_q1p0x_seed42_temp1p5_c8_r1.json results/causal_hybrid_n4_obs3_fixed_deepseek-chat_reasoned_strict_q1p0x_seed42_temp0p7_c8_r1.json results/causal_hybrid_n4_obs3_fixed_deepseek-chat_reasoned_strict_q1p0x_seed42_temp1p5_c8_r1.json results/causal_hybrid_n4_obs3_fixed_deepseek-chat_reasoned_strict_sysenhanced_q1p0x_seed42_temp1p5_c8_r1.json --labels "Baseline T0.7" "Baseline T1.5" "Baseline T1.5 (SysEnhanced)" "Strict T0.7" "Strict T1.5" "Strict T1.5 (SysEnhanced)" --out-bars figs/causal_six_compare_seed42_bars.png --out-scatter figs/causal_six_compare_seed42_scatter.png
```

- 提示：如只需柱状图，省略 `--out-scatter`；当前报告仅引用 `figs/causal_six_compare_seed42_bars.png`。
