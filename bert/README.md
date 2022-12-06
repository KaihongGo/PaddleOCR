KIE 任务 ops创建 参数没有用到 global参数，虽然进行了参数更新。

paddlenlp >=2.4.1, <=2.4.2, commit: 4ccf80df release/2.4
paddleocr commit: 6eb5d3e release/2.6

pytest patch argparse

```python
@pytest.fixture
def preprocess(mocker):
    mocker.patch(
        "tools.program.ArgsParser.parse_args",
        return_value=program.ArgsParser().parse_args(
            [
                "--config",
                "configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml",
            ]
        ),
    )
    config, device, logger, vdl_writer = program.preprocess(is_train=True)
    seed = config["Global"]["seed"] if "seed" in config["Global"] else 1024
    set_seed(seed)
    return config, device, logger, vdl_writer
```