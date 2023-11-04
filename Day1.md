# 创建趋动云云资源

这一步会因为不同公司而不同因此不放上了， 贴个图

![image-20231104102248266](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041205993.png)

# 配置开发环境

- 打开JuypterLab

![image-20231104102411916](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041205792.png)

- 创建新的ipylab，打开终端，输入以下指令进行换源

```shell
apt-get update&&apt-get install unzip
git config --global url."https://gitclone.com/".insteadOf https://
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
python3 -m pip install --upgrade pip
```

![image-20231104103421361](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206815.png)

- 下载模型，修改requirements

```shell
git clone https://github.com/THUDM/ChatGLM3.git
cd ChatGLM3
```

删除后的requirements

```txt
protobuf
transformers==4.30.2
cpm_kernels
gradio==3.39
mdtex2html
sentencepiece
accelerate
sse-starlette
streamlit>=1.24.0

```

安装依赖
```shell
pip install -r requirements.txt
```

修改web_demo.py web_demo2.py前几行中的tokenizer中的路径，修改为

```shell
../../pretrain
```

同时修改web_demo.py中最后一行的启动代码

```shell
demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7000 )
```

在界面右边添加端口7000，即刚才填入的server_port

![image-20231104105149178](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206185.png)

- 运行模型，启动gradio界面

```shell
python web_demo.py
```

出现错误提示, 原因是漏掉了斜杠

```shell
Traceback (most recent call last):
  File "/gemini/code/ChatGLM3/web_demo.py", line 6, in <module>
    tokenizer = AutoTokenizer.from_pretrained("../..pretrain", trust_remote_code=True)
  File "/root/miniconda3/lib/python3.9/site-packages/transformers/models/auto/tokenization_auto.py", line 643, in from_pretrained
    tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
  File "/root/miniconda3/lib/python3.9/site-packages/transformers/models/auto/tokenization_auto.py", line 487, in get_tokenizer_config
    resolved_config_file = cached_file(
  File "/root/miniconda3/lib/python3.9/site-packages/transformers/utils/hub.py", line 417, in cached_file
    resolved_file = hf_hub_download(
  File "/root/miniconda3/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py", line 112, in _inner_fn
    validate_repo_id(arg_value)
  File "/root/miniconda3/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py", line 166, in validate_repo_id
    raise HFValidationError(
huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: '../..pretrain'.
```

- 运行完成，但没有外部打开链接无法显示

```shell
/gemini/code/ChatGLM3/web_demo.py:89: GradioDeprecationWarning: The `style` method is deprecated. Please set these arguments in the constructor instead.
  user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
Running on local URL:  http://0.0.0.0:7000

To create a public link, set `share=True` in `launch()`.
```

![image-20231104105901544](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206570.png)

找到外部访问链接后

![image-20231104110152960](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206038.png)

仍然无法打开

![image-20231104110210321](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206385.png)

![image-20231104110224257](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206117.png)

经过群友提示，应该是内存正在交换，不足以启动，重启一次即可

![image-20231104111011710](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206738.png)

![image-20231104111227801](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206961.png)

```shell
streamlit run web_demo2.py
```

![image-20231104111539848](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206641.png)

打开外部链接时，仍然无法加载模型，多次重启进程，重启环境后后仍然无法打开。 

初步估计是GPU被写入后显存没有释放，尝试直接运行demo2

等待释放完毕
![image-20231104115642122](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206107.png)

再次运行
![image-20231104115857282](https://raw.githubusercontent.com/most-silence/picgo_image/main/image_note/202311041206261.png)

未找到原因
任务一结束