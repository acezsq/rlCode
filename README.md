# rlCode

本代码仓库主要是修改cleanrl的rl算法实现以适配自定义gym环境。使用本代码仓库更容易和大家自己定义的gym环境对接。

修改的主要内容：

（1）更改cleanrl中多个环境并行采集数据的实现为单环境采集（为了适配我自己的gym任务环境）

（2）将gymnasium改为的gym（我自己的环境之前是基于gym实现的）

（3）去除部分我认为暂时没必要的配置项

（4）增加少量控制台打印训练进度信息展示

已完成修改的常用算法列表：
 
| rl算法 | cleanrl | my | my代码是否验证 |
| :---: | :---: | :---: | :---: |
| ppo | [ppo](https://github.com/acezsq/rlCode/blob/main/ppo.py) | [ppo_new](https://github.com/acezsq/rlCode/blob/main/ppo_new.py) | ✅ |
| ppo_atari | [ppo_atari](https://github.com/acezsq/rlCode/blob/main/ppo_atari.py) | [ppo_atari_new](https://github.com/acezsq/rlCode/blob/main/ppo_atari_new.py) | ✅ |
| dqn | [dqn](https://github.com/acezsq/rlCode/blob/main/dqn.py) | [dqn_new](https://github.com/acezsq/rlCode/blob/main/dqn_new.py) | ✅ |
| dqn_atari | [dqn_atari](https://github.com/acezsq/rlCode/blob/main/dqn_atari.py) | [dqn_atari_new](https://github.com/acezsq/rlCode/blob/main/dqn_atari_new.py) | ✅ |
| sac_atari | [sac_atari](https://github.com/acezsq/rlCode/blob/main/sac_atari.py) | [sac_atari_new](https://github.com/acezsq/rlCode/blob/main/sac_atari_new.py) | ✅ |
| ..... |  |  |  |
| ..... |  |  |  |


ppo和dqn普通版本目前已验证经过我的修改之后没有大问题，可以直接用；atari版本也可以跑通，但是需要注意需要将输入的图像堆叠成四个维度batch * channel * height * width（比如将1 * 4 * 84 * 84这种）。
训练的对比效果图如下：

![dqn vs dqn_new.png](https://github.com/acezsq/rlCode/blob/main/pic/dqn%20vs%20dqn_new.png)
![ppo vs ppo_new](https://github.com/acezsq/rlCode/blob/main/pic/ppo%20vs%20ppo_new.png)

致谢cleanrl：https://github.com/vwxyzjn/cleanrl
