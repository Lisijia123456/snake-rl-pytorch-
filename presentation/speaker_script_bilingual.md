# Speaker script · 15 slides · about 20 seconds per slide

Chinese is the intended spoken version. English is provided as a reference.

## Slide 1

**中文**：大家好，我是李思佳。今天介绍 DTE2502 神经网络课程作业二：把 Snake 项目的 Deep Q-Learning Agent 从 TensorFlow/Keras 移植到 PyTorch。重点是 agent.py 的实现、训练流程，以及最终模型在障碍棋盘上的表现。

**English**: Hello, I am Li Sijia. This presentation covers Assignment 02 for DTE2502: porting the Snake Deep Q-Learning agent from TensorFlow/Keras to PyTorch. I will focus on the agent implementation, training pipeline, and measured policy results.

## Slide 2

**中文**：这一部分介绍项目本身：我们用 PyTorch CNN 学习贪吃蛇的动作选择，沿用 v17.1 的模型配置，并保留运行游戏、训练和可视化所需的代码。项目提供训练好的检查点和可复现的运行流程。

**English**: This project uses a PyTorch CNN to choose moves in Snake. It follows the v17.1 model configuration and includes the code needed to run the game, train the agent, and visualize its policy, along with a trained checkpoint.

## Slide 3

**中文**：环境是在十乘十棋盘上玩贪吃蛇。状态由最近两帧组成，动作有上下左右四种，非法动作会被屏蔽。数据文件保存四万零八条棋盘记录，每张棋盘有八个障碍格，其中有两万三千三百三十二种不同布局。

**English**: The environment is Snake on a ten-by-ten board. The state contains the last two frames, and the agent chooses among four directions while illegal moves are masked. The dataset stores 40,008 board records with eight obstacle cells per board.

## Slide 4

**中文**：DQN 使用经验回放和目标网络稳定学习。每次从最多八万条经验中抽取六十四条，使用 Huber 损失和 RMSprop 更新。目标网络每五百次迭代同步，探索率逐步下降到百分之一。

**English**: DQN uses replay memory and a target network for stable learning. Each update samples 64 transitions from a buffer of up to 80,000. We use Huber loss and RMSprop, sync the target network every 500 iterations, and decay exploration to one percent.

## Slide 5

**中文**：项目围绕 v17.1 的 DQN 流程整理：保留智能体、环境、经验回放、训练和可视化相关代码，并移除了其它算法和过时的模型版本，让目录结构更精简。

**English**: The project is organized around the v17.1 DQN workflow. It keeps the agent, environment, replay buffer, training, and visualization code, while removing unrelated algorithms and outdated model versions to keep the project compact.

## Slide 6

**中文**：一次训练迭代先让一百二十八个游戏并行采取动作，再把状态转移放入回放缓存，然后抽取一个批次更新网络。每五百次迭代评估八局、同步目标网络并保存检查点，循环共进行二十万次。

**English**: Each iteration advances 128 parallel games, stores transitions, and performs one minibatch update. Every 500 iterations, the script evaluates eight games, syncs the target network, and saves a checkpoint. Training ran for 200,000 iterations.

## Slide 7

**中文**：网络结构由 JSON 配置构建。输入是两帧十乘十棋盘，经过三层卷积、一个六十四单元全连接层，最后输出四个动作价值。模型共有一百二十二零六十八个可训练参数。

**English**: The CNN is built from the JSON configuration. It takes two ten-by-ten frames, applies three convolutional layers and a 64-unit dense layer, then outputs four action values. It has 122,068 trainable parameters.

## Slide 8

**中文**：移植时重点对齐框架细节：Keras 的通道最后格式转换为 PyTorch 的通道优先格式；显式设置 Glorot 初始化、RMSprop 参数和 Huber 损失；权重以 PyTorch state_dict 格式保存。

**English**: The port aligns important framework details: Keras channels-last data becomes PyTorch channels-first data; Glorot initialization, RMSprop settings, and Huber loss are set explicitly; weights are stored as a PyTorch state dictionary.

## Slide 9

**中文**：这里展示输入预处理和动作选择。环境与回放仍使用 NumPy 数组，只在智能体内部把棋盘归一化并转换为 NCHW 张量。预测时关闭梯度，只在合法动作中选择 Q 值最大的动作。

**English**: This slide shows preprocessing and action selection. The environment and replay buffer remain NumPy-based. Inside the agent, boards are normalized and converted to NCHW tensors. Inference disables gradients and selects the highest-valued legal action.

## Slide 10

**中文**：每次更新先从经验缓存取样，计算目标网络对下一状态的价值，再结合奖励和终止标记形成贝尔曼目标。训练时仅更新实际采取动作对应的 Q 值，其余动作的预测作为固定目标。

**English**: For an update, the agent samples transitions, evaluates next states with the target network, and forms the Bellman target from reward, discount, and terminal status. Only the Q value for the chosen action is updated.

## Slide 11

**中文**：其它改动尽量保持局部：训练脚本支持短跑和完整训练，环境去掉未使用的 TensorFlow 引用并兼容新 NumPy，GIF 输出使用 Pillow。README 列出了环境准备、训练、绘图和策略演示命令。

**English**: Changes elsewhere stay focused: the training script supports quick and full runs, the environment drops an unused TensorFlow import and handles newer NumPy, and GIF export uses Pillow. The README documents setup and run commands.

## Slide 12

**中文**：移植遇到的挑战主要是张量布局和默认参数差异。另一个细节是终止状态可能没有合法动作，如果直接把负无穷乘以零会得到无效数值，所以实现中先清理这个目标值。

**English**: The main porting challenges were tensor layout and differing framework defaults. We also guard against invalid targets when terminal states have no legal actions, because multiplying negative infinity by zero would otherwise produce NaN.

## Slide 13

**中文**：结果页展示二十万次迭代检查点的学习曲线。最后一次八局评估的平均奖励是十八点八八、平均长度二十一点八八；最后一万次迭代的平均奖励约十五点一二。每次只评估八局，因此指标会有波动。

**English**: The plot is from the 200,000-iteration checkpoint. The last eight-game evaluation logged mean reward 18.88 and mean length 21.88. Across the final 10,000 iterations, the means were 15.12 and 18.12; small evaluations are noisy.

## Slide 14

**中文**：这三张图取自最终检查点的三次独立障碍棋盘游戏，得分分别是十三、二十六和十八。它们展示的是实际策略运行画面，不代表未见棋盘或空棋盘上的泛化测试。

**English**: These are frames from three separate games using the final checkpoint on generated obstacle boards, with scores 13, 26, and 18. They are representative rollouts, not held-out or empty-board generalization tests.

## Slide 15

**中文**：总结一下，我完成了 v17.1 智能体的 PyTorch 实现、训练及策略可视化，并整理了可复现说明。后续可以尝试 Double DQN 或优先经验回放。公开仓库地址需在项目发布后补入。谢谢。

**English**: In summary, I implemented and trained the v17.1 PyTorch agent and visualized its policy. Double DQN and prioritized replay are possible next steps. The public repository URL should be added after publishing. Thank you.
