# WidowXAI training-time RTC

## 何を変更するか

参照は `real-time-chunking/src/model.py` の `simulated_delay` 分岐。
新しいadapter model、層、学習パラメータ、raw dataの再変換は追加しない。
既存のaction timestep入力を `[B]` に加えて `[B,H]` で渡せるようにする。
DiT blockとMoTのtoken別modulationは既に対応しているので、それを利用する。
state dictのキーと重みのshapeは通常モデルと同じ。

学習時RTCは学習済み重みを必要とする方式であり、既存checkpointに対する推論時RTC
(`guidance_method: rtc`) とは別。学習時RTCの推論 (`guidance_method: train`) では
VJPや追加のguidance lossを計算せず、遅延中に実行されるprefixを固定する。

## 現在のraw data → model入力

このリポジトリの学習入力は、既にLeRobot形式に変換されたデータセット。
`meta/info.json`、episode metadata、Parquetのaction/state、カメラ動画を使う。
元のHDF5/MCAP等からLeRobotへ変換する汎用変換器は今回追加しない。

2手首の既定設定 `configs/data/widowxai_2wrist_vertical.yaml` の流れは次のとおり。

1. `BaseLerobotDataset` が先頭時点を基準に、33時点の画像/stateと32個のactionを取得。
   `global_sample_stride=1`。episode末尾を超える要素は複製し、padding maskを付ける。
2. `FastWAMProcessor` が両手首画像を各240×320へ変換。
   `MaskedRelativeJointTransform` が12関節のactionから先頭stateを引く。
   gripperの次元6・13は絶対値を維持する。
3. action/stateを保存済み `dataset_stats.json` でz-score正規化し、14次元へmerge。
   stateは相対値に変えず、state用統計で正規化する。
4. `RobotVideoDataset` が画像を0,4,...,32の9時点に間引く。actionは32個のまま。
   各手首を192×256へaspect-fitし、左右に32pxずつgrayを追加。
   左手首を上、右手首を下に配置して384×320の画像を作り、[-1,1]へ正規化。
5. instructionに対応する保存済みT5 embeddingを読み込む。
   バッチ入力は `video:[B,3,9,384,320]`, `action:[B,32,14]`,
   `proprio:[B,32,14]`, `context:[B,128,4096]` とpadding mask。
6. モデルの `build_inputs` がVAEで動画をlatentへ変換し、先頭proprioをtext contextへ追加。
   通常学習ではvideo/actionへ独立にノイズとtimestepをサンプルしてflowを学習する。

## RTCで増える入力加工とloss

データセットとprocessorは変更せず、モデルの `training_loss` 内に次を追加する。

- `D=model.training_rtc.simulated_delay` として `d∈{0,...,D-1}` をサンプル。
  参照実装と同じ `p(d)∝exp(-d)`。`D=8` なら最大遅延は **7** ステップ。
- `sigma=timestep/1000` の既存スケールを使う。
  prefix `[0:d]` はノイズなし教師actionとtimestep=0。
  suffix `[d:H]` は通常の `(1-sigma)*action + sigma*noise` と元のtimestep。
  参照のclean時刻1はFasterWAMでは0なので、向きを変換している。
- targetは従来どおり `noise-action`。action lossをprefixとpadding以外だけで平均する。
  次元方向の平均、schedulerのサンプル重み、video lossは従来のまま。
- episode末尾ではprefixを短くし、実データのsuffixを最低1個残す。
  全actionがpaddingなら、そのサンプルのaction lossは0。
- `d=0`も学習するため、初回のprefixなし推論を同じモデルで行える。

実行時は未実行旧chunkを現在のqpos基準で再相対化・正規化し、
先頭 `min(inference_delay_steps, len(previous_actions))` 個を各denoise stepで固定する。
更新後も再固定するので、返却chunkのprefixは入力と一致する。
`one_pass_future_cache` と `joint_rollout` の両方に対応する。
遅延予測が学習範囲を超えた場合は警告する。予測値を学習範囲内へ黙って丸めない。
全horizonを固定する要求は生成対象がなくなるため拒否する。

この実装は参照と同じく教師軌跡をprefixとして学習する。
旧policyの予測誤差、実機フィルタによる変更、可変遅延下のタスク性能は別途評価が必要。
`D` は測定した実機の推論往復遅延をカバーするよう選ぶ。

## training configの切替

| 設定 | ファイル / 値 |
|---|---|
| 共通の学習設定 | `configs/train.yaml` |
| 既存2手首の実験設定 | `configs/task/widowxai_fasterwam_2wrist_vertical_384_1e-4.yaml` |
| 学習時RTCの実験設定 | `configs/task/widowxai_fasterwam_2wrist_training_rtc.yaml` |
| データ・processor | `configs/data/widowxai_2wrist_vertical.yaml` |
| モデル・既定RTC無効 | `configs/model/fasterwam.yaml`: `training_rtc.simulated_delay: null` |

通常学習は既存taskのまま。RTC学習は以下で切り替える。

```bash
python scripts/train.py task=widowxai_fasterwam_2wrist_training_rtc --cfg job --resolve
# 最大遅延を15ステップへ変更
python scripts/train.py task=widowxai_fasterwam_2wrist_training_rtc \
  model.training_rtc.simulated_delay=16 --cfg job --resolve
```

`--cfg job --resolve` はconfig表示のみで、学習・モデルロードを行わない。
既存taskでも `model.training_rtc.simulated_delay=8` で有効化できる。
`null` なら通常学習、`1` なら常にd=0でprefixなしの学習となる。

## 別GPUサーバーで短い学習を起動する

同じbranch/commitを別サーバーへ取り込み、既存のFasterWAM用Python環境を有効にする。
PyTorch・Accelerate・DeepSpeedなどは既存のインストール済み環境を使う。
起動スクリプトはジョブ投入やpackage/modelのdownloadを行わない。
Wanのローカル重み、LeRobotデータ、正規化統計、taskのtext cacheが必要。

```bash
cd /path/to/FasterWAM
export TRAIN_DATASET=/path/to/real_widowxai_success_only
export VAL_DATASET=/path/to/real_widowxai_test
export NORM_STATS=/path/to/matching/dataset_stats.json
export TEXT_CACHE=/path/to/text_embeds_cache/widowxai
export MODEL_ROOT=/path/to/checkpoints
export OUTPUT_ROOT=/path/to/new/training_rtc_runs
export HF_HOME=/path/to/writable/hf_cache

# 任意：通常学習済みFasterWAMから追加学習する場合。
# .ptのweightsだけをロードし、optimizer/scheduler/stepは新しく開始する。
export INIT_CHECKPOINT=/path/to/ordinary_run/checkpoints/weights/step_000960.pt

# configと入力パスを確認。独立したrun directoryを作るが学習はしない。
bash scripts/train_widowxai_training_rtc.sh --dry-run

# 1 GPU、batch=1、5 optimizer steps。GPUメモリ量に応じてNPROCを増やす。
NPROC=1 MAX_STEPS=5 BATCH_SIZE=1 \
  bash scripts/train_widowxai_training_rtc.sh
```

`INIT_CHECKPOINT` が未指定なら、既存taskと同様にWan video expertの事前学習重みと
ランダム初期化action expertから開始する。action backboneの初期化を再利用する場合は
`ACTION_DIT_PATH=/path/to/SparseActionDiT_...pt` を指定する。
初期化の違いは実験比較に影響するので、混同しない。
通常学習のoptimizerを含む再開ではなく、RTCへの追加学習にはweightsの `.pt` を推奨する。

スクリプトの既定は2手首・384×320・bf16・ZeRO-2 CPU offload。
5Bモデルなのでbatch=1でも小容量GPUで動く保証はない。
必要なら `NPROC=8` など、別サーバーの利用可能なGPU数を指定する。
CPU optimizerの初回buildが必要な環境では、既存の
`python scripts/prebuild_ds_ops.py` をそのサーバーで先に実行する。

主要環境変数は `SIMULATED_DELAY`, `SEED`, `NUM_WORKERS`, `RTC_PYTHON`, `RUN_ID`。
追加のHydra overrideはスクリプトの末尾引数で渡せる。
長い実験では `MAX_STEPS=1600` と `num_epochs=28` などを明示する。
実機・学習性能の比較は同じデータ、初期checkpoint、seedで行う。

各実行は新しいrun directoryに次を保存する。

- `resolved_config.yaml` / `config.yaml`：実行設定。
- `launch_manifest.json`：code commit/status、seed、初期checkpoint hash、
  normalization hash、dataset metadataのhash、text cacheの場所。
  dataset hashはmetadataのfingerprintで、全動画・Parquetのhashではない。
- `command.txt` / `train.log`：起動コマンドと学習ログ。
- `checkpoints/weights/step_*.pt` と学習state。

同じ`RUN_ID`の再利用は拒否し、過去stateの自動削除も無効にする。
text cacheがない場合は既存のprecompute手順で別途準備する。

## 学習済みRTCモデルを非同期推論で使う

学習runのconfigとcheckpoint、同じnormalization statsを使う。
サーバーは学習configでRTCが有効だったrunに限り`train`を許可する。

```bash
PYTHONPATH=src python scripts/serve_widowxai_policy.py \
  --run-dir /path/to/training_rtc_run \
  --model-base-path "$MODEL_ROOT" \
  --text-embedding-cache-dir "$TEXT_CACHE" \
  --serving-config configs/serve/train.yaml
```

`joint_rollout` は `--action-infer-mode joint_rollout` を追加する。
ロボットPCは既存SSHトンネルを使い、sbr-wm-deploy側で以下を指定する。

```bash
cd /path/to/sbr-wm-deploy/trossen_ai_aloha
python scripts/eval_policy_remote_inf.py \
  -r asratec_xai_stationary_inference \
  --host 127.0.0.1 --port 8000 \
  --inference-config config/inference_train.yaml
```

既存 `config/inference.yaml` を使う場合は `mode: async`, `guidance_method: train`。
`mode`は実行方式なので`train`には変更しない。
`train`非対応のサーバーへの要求はロボット初期化前に失敗する。
robot PCでは学習しない。

## ローカル検証

```bash
PYTHONPATH=src python scripts/test_training_rtc.py -v
PYTHONPATH=src python scripts/test_rtc.py -v
PYTHONPATH=src python scripts/test_chunk_protocol.py \
  --client-scripts /path/to/sbr-wm-deploy/trossen_ai_aloha/scripts -v
```

新規テストは実際のSparseActionDiT/SparseMoT/WanVideoDiTを小さくしてCPU上で使い、
VAEだけを軽量な変換に置き換える。3回のoptimizer update、checkpoint save/load、
prefix/paddingのloss mask、token timestep、両推論経路のprefix固定、通常経路の回帰、
servingと通信契約を検証する。raw dataや5Bモデルを使った学習smokeとは区別する。
