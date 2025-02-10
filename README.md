![ATRI.jpg](img%2FATRI.jpg)

---
## 简介
简体中文 · [English](./README_EN.md) 
- My Dream Moments 是一个基于大型语言模型（LLM）的情感陪伴程序，能够接入微信，提供更真实的情感交互体验。内置了 Atri-My dear moments 的 prompt，并且解决了传统人机对话程序一问一答的死板效果，提供沉浸式角色扮演和多轮对话支持。项目名字来源是运行在该程序上的第一个智能体的原作名称+项目价值观混合而来。
- 推荐使用DeepSeek V3 模型。<br>
![demo.png](img%2Fdemo.png)
---
## 声明
- 本项目仅用于交流学习，LLM发言不代表作者本人立场。prompt所模仿角色版权归属原作者。任何未经许可进行的限制级行为均由使用者个人承担。
---

## 已实现的功能
- [x] 对接微信，沉浸式角色扮演
- [x] 聊天分段响应，消除人机感
- [x] 多轮对话
- [x] 多用户支持
- [x] 由 DeepSeek R1 利用游戏文本生成的 Prompt
- [x] 无需联网的时间感知

---

## 待实现功能，期待您的加入
- [ ] 定时任务与智能定时任务
- [ ] 利用 8B 小模型实时生成记忆并定期整理，实现持久记忆
- [ ] WebUI，方便不理解代码的用户配置项目
- [ ] 负载均衡
- [ ] Releases

---

## 如何运行项目

### 这里是QQ分支~

#### Windows

前期准备:

1. **QQ小号**  
   - NapCat是目前较为安全的机器人后端

2. **DeepSeek API Key**  
   - 硅基流动：[获取 API Key（15元免费额度）](https://cloud.siliconflow.cn/i/nTXFQjY5)

在[Releases · NapNeko/NapCatQQ](https://github.com/NapNeko/NapCatQQ/releases/)下载[**Win64无头**]一键包。解压后运行napcat.bat，根据提示扫码登陆。

<img src="https://i.ibb.co/35xzZMjv/image-20250210150629365.png" style="zoom:33%;" />

之后在终端找到带有token的webui链接。建议修改默认token(密码).

进入你想安装 FloraBot 的目录中, 然后运行以下命令(请使用 PowerShell 来运行):（[FloraBotTeam/FloraBot-Installer: FloraBot 的一键安装脚本](https://github.com/FloraBotTeam/FloraBot-Installer)）

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/FloraBotTeam/FloraBot-Installer/main/WindowsInstaller.ps1" -OutFile "WindowsInstaller.ps1"; powershell -File WindowsInstaller.ps1; Remove-Item WindowsInstaller.ps1
```

如果网络不好，运行：

```powershell
Invoke-WebRequest -Uri "https://github.moeyy.xyz/https://raw.githubusercontent.com/FloraBotTeam/FloraBot-Installer/main/WindowsInstaller.ps1" -OutFile "WindowsInstaller.ps1"; powershell -File WindowsInstaller.ps1; Remove-Item WindowsInstaller.ps1
```



根据提示完成后，启动脚本为 Run.bat, 运行该脚本即可启动 FloraBot。配置目录下的`Config.json`。把`ConnectionType`的值改为**`WebSocket`**,这样方便配置.在`BotID`把0换为机器人QQ号,在`Administrator`把0换为管理员QQ号,可以写多个.

在[umaru-233/My-Dream-Moments at QQ-FloraBotPlugin](https://github.com/umaru-233/My-Dream-Moments/tree/QQ-FloraBotPlugin)点击Code - Download ZIP,解压后的插件放在Plugins目录.最后的目录结构类似:

```
D:\Apps\FloraBot\FloraBot\Plugins\My-Dream-Moments-QQ-FloraBotPlugin\Main.py...
```

插件目录有Plugin.json.在DeepSeekApiUrl, DeepSeekApiKey, DeepSeekModel键值对配置自己的api即可.其他配置按需要调整.

回到NapCat的webui.在 网络配置 栏新建一个WebSocket客户端.默认端口和flora一样都是3003不用改动.

<img src="https://i.ibb.co/TB551Gk5/image.png" style="zoom: 33%;" />

启动ws客户端后,重新运行florabot的Run.bat.你应该可以看见类似的输出:

```
正在加载插件, 请稍后...
正在加载插件 MyDreamMoments ...
框架连接方式为: WebSocket
MyDreamMoments 加载成功
框架已通过 WebSocket 连接, 连接ID: 1
```

这样就可以在群聊或者私聊使用命令和机器人聊天了.具体可以发送

```
/帮助
/帮助 MyDreamMoments
```

查看. 群聊内也可以使用@进行聊天.

目前qq插件尚在开发中,本文仅供学习参考~

原文链接:[使用NapCat与FloraBot部署对接DeepSeek的QQ聊天机器人（老婆） | 独立遗世的理想乡](https://mondavalon.github.io/posts/8bfef174/)

### 微信分支

#### 1. 前期准备
1. **备用手机/安卓模拟器**  
   - 微信电脑端登录必须有一个移动设备同时登录，因此不能使用您的主要设备。
   
2. **微信小号**  
   - 需要注册时间较久的微信号（新注册的账号可能无法登录为 bot）。如果扫码后报错，请检查此项。

3. **DeepSeek API Key**  
   - 推荐使用：[获取 API Key（15元免费额度）](https://cloud.siliconflow.cn/i/aQXU6eC5)

---

#### 2. 部署项目
1. **克隆本仓库**  
   ```bash
   git clone <仓库地址>
2. **安装依赖**  
   ```bash
   pip install -r requirements.txt
3. **配置<code>config.py</code>**  
修改<code>DEEPSEEK_API_URL</code>和<code>DEEPSEEK_API_KEY</code>。
按需调整<code>MAX_TOKEN</code>、<code>TEMPERATURE</code>和<code>MODEL</code>。
4. 运行<code>bot.py</code>
   ```bash
   python bot.py
   ```
#### 3. 如何使用
1. **项目运行后，根目录下会生成 <code>QR.png</code>。**
   - 使用您想作为 bot 的微信号扫码登录。
   - 如果控制台没有响应，请重启项目以刷新二维码并重新登录。
2. **当控制台提示 <code>Start auto replying.</code> 时，表示程序已成功运行。您可以在微信上与 bot 对话测试效果。**

### 如果您想修改prompt
- 项目根目录下的 <code>prompt.md</code> 可以编辑，修改后重启项目生效。
- 注意：请不要修改与反斜线 <code> \ </code>相关的 prompt，因为它们被用于分段回复消息。
## 赞助
此项目欢迎赞助。您的支持对我非常重要！
- 赞助用户如需远程部署或定制 prompt，请联系我。
- E-Mail: yangchenglin2004@foxmail.com 
- Bilibili: [umaru今天吃什么](https://space.bilibili.com/209397245)
- 感谢您的支持与使用！!<br>
![qrcode.jpg](img%2Fqrcode.jpg)
