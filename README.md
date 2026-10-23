# XR1 submission for RoboSynChallenge

This repository contains one complete, scanner-visible RoboSynChallenge policy at
`policy/xr1_submission/`. The adapter, evaluator entry point, inference runtime,
installer, downloader, and ten-task mapping are all inside that directory.

The checkpoints are hosted in
[`87361Yly/robosyn-xr1-sim10`](https://huggingface.co/87361Yly/robosyn-xr1-sim10)
at immutable revision `fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4`.

## Repository layout

```text
robosyn-xr1-submission/
├── policy/
│   └── xr1_submission/
│       ├── eval.sh
│       ├── deploy_policy.py
│       ├── deploy_policy.yml
│       ├── install.sh
│       ├── download_checkpoint.sh
│       ├── task_mapping.json
│       ├── requirements-inference.txt
│       ├── runtime/
│       └── README.md
├── LICENSE
└── README.md
```

See [`policy/xr1_submission/README.md`](policy/xr1_submission/README.md) for
installation, checkpoint download, runtime behavior, and configuration details.

## Install into the organizer checkout

Run these commands from the directory that contains the official
`RoboSynChallenge/` checkout:

```bash
git clone https://github.com/87361/robosyn-xr1-submission.git
cp -a robosyn-xr1-submission/policy/xr1_submission RoboSynChallenge/policy/

cd RoboSynChallenge
bash policy/xr1_submission/install.sh
bash policy/xr1_submission/download_checkpoint.sh

# Activate the organizer's simulator environment first.
bash policy/xr1_submission/eval.sh click_bell random
```

## Ten task entry points and checkpoints

Each command below is run from the official `RoboSynChallenge/` root after the
policy has been copied and the checkpoints have been downloaded.

| Task | Evaluation entry point | Immutable checkpoint URL |
| --- | --- | --- |
| `click_bell` | `bash policy/xr1_submission/eval.sh click_bell random` | [`click_bell_160k_ft3k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/click_bell_160k_ft3k) |
| `handle_basket` | `bash policy/xr1_submission/eval.sh handle_basket random` | [`handle_basket_140k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/handle_basket_140k) |
| `water_pouring` | `bash policy/xr1_submission/eval.sh water_pouring random` | [`water_pouring_160k_ft5k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/water_pouring_160k_ft5k) |
| `table_rearrangement` | `bash policy/xr1_submission/eval.sh table_rearrangement random` | [`table_rearrangement_160k_ft5k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/table_rearrangement_160k_ft5k) |
| `items_handover` | `bash policy/xr1_submission/eval.sh items_handover random` | [`items_handover_180k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/items_handover_180k) |
| `drawer_open_place` | `bash policy/xr1_submission/eval.sh drawer_open_place random` | [`drawer_open_place_160k_ft10k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/drawer_open_place_160k_ft10k) |
| `mixer_operating` | `bash policy/xr1_submission/eval.sh mixer_operating random` | [`mixer_operating_terminal_press_ft4k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/mixer_operating_terminal_press_ft4k) |
| `item_assembly` | `bash policy/xr1_submission/eval.sh item_assembly random` | [`item_assembly_160k_ft10k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/item_assembly_160k_ft10k) |
| `manipulate_pipette` | `bash policy/xr1_submission/eval.sh manipulate_pipette random` | [`manipulate_pipette_160k_ft20k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/manipulate_pipette_160k_ft20k) |
| `sample_loading` | `bash policy/xr1_submission/eval.sh sample_loading random` | [`sample_loading_160k_ft20k`](https://huggingface.co/87361Yly/robosyn-xr1-sim10/tree/fdc1c2f53a0eeb61b02459ef4fca41c7e2aa0dc4/sample_loading_160k_ft20k) |
