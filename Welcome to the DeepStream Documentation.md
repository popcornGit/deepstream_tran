# NVIDIA DeepStream Overview
DeepStream是一个流媒体分析工具包，用于构建人工智能驱动的应用程序。
它将来自USB/CSI相机的流数据、来自文件的视频或RTSP流作为输入，并使用人工智能和计算机视觉从像素中生成见解，以更好地了解环境。
DeepStream SDK可以作为许多视频分析解决方案的基础层，如了解智能城市中的交通和行人、医院中的健康和安全监测、零售中的自助结账和分析、检测制造设施中的组件缺陷等。
![image](https://github.com/popcornGit/deepstream_tran/assets/48575896/f8eee78d-d2bc-40d9-b53c-898163862319)

DeepStream通过Python绑定支持C/C++和Python中的应用程序开发。
为了更容易上手，DeepStream提供了C/C++和Python中的几个参考应用程序。
请参阅C/C++示例应用程序源详细信息和Python示例应用程序和绑定源详细信息部分，以了解有关可用应用程序的更多信息。
有关一些DeepStream参考应用程序示例，请参阅NVIDIA-IOT GitHub页面。

核心SDK由几个硬件加速器插件组成，这些插件使用加速器，如VIC、GPU、DLA、NVDEC和NVENC。
通过在专用加速器中执行所有计算量大的操作，DeepStream可以实现视频分析应用程序的最高性能。
DeepStream的关键功能之一是边缘和云之间的安全双向通信。
DeepStream提供了几种开箱即用的安全协议，如使用用户名/密码的SASL/Plain身份验证和双向TLS身份验证。
要了解有关这些安全功能的更多信息，请阅读物联网章节。
要了解有关双向功能的更多信息，请参阅本指南中的双向消息部分。

DeepStream构建在CUDA-X堆栈中的几个NVIDIA库之上，如CUDA、TensorRT、NVIDIA®Triton™ 推理服务器和多媒体库。
TensorRT加速了NVIDIA GPU上的人工智能推理。
DeepStream在DeepStream插件中对这些库进行了抽象，使开发人员无需学习所有单独的库即可轻松构建视频分析管道。

DeepStream针对NVIDIA GPU进行了优化；应用程序可以部署在运行Jetson平台的嵌入式边缘设备上，或者可以部署在更大的边缘或数据中心GPU（如T4）上。
DeepStream应用程序可以使用NVIDIA容器运行时部署在容器中。
这些容器可在NGC、NVIDIA GPU云注册表上获得。要了解有关使用Docker部署的更多信息，请参阅Docker容器一章。
DeepStream应用程序可以使用GPU上的Kubernetes在边缘进行编排。
NGC上提供了部署DeepStream应用程序的Helm图示例。

## DeepStream Graph Architecture

DeepStream是一个使用开源GStreamer框架构建的优化图架构。
下图显示了一个典型的视频分析应用程序，从输入视频到输出见解。
所有单独的块都是使用的各种插件。
底部是在整个应用程序中使用的不同硬件引擎。
优化的内存管理，插件之间的零内存拷贝和各种加速器的使用确保了最高的性能。
![image](https://github.com/popcornGit/deepstream_tran/assets/48575896/6960529f-f7ea-43de-b795-5f4dbff8a3a7)

DeepStream以GStreamer插件的形式提供构建块，可用于构建高效的视频分析管道。有20多个插件是针对各种任务进行硬件加速的。

流式数据可以通过RTSP通过网络传输，也可以直接来自本地文件系统或相机。流是使用CPU捕获的。
一旦帧在存储器中，它们就被发送以使用NVDEC加速器进行解码。解码插件名为Gst-nvvideo4linux2。

在解码之后，存在可选的图像预处理步骤，其中可以在推断之前对输入图像进行预处理。
预处理可以是图像脱蜡或颜色空间转换。
Gst nvdewarper插件可以对鱼眼或360度相机的图像进行反锐化。
Gst nvvideoconvert插件可以对帧进行颜色格式转换。
这些插件使用GPU或VIC（视觉图像合成器）。

下一步是批量处理帧以获得最佳推理性能。批处理是使用Gst nvstreammux插件完成的。

一旦帧被批处理，它就被发送用于推理。
可以使用TensorRT（NVIDIA的推理加速器运行时）进行推理，也可以使用Triton推理服务器在TensorFlow或PyTorch等原生框架中进行推理。
使用Gst nvinfer插件执行原生TensorRT推断，使用Triton的推断使用Gst envinferserver插件执行。
推理可以使用GPU或DLA（深度学习加速器）进行Jetson AGX Orin和Orin NX。

推断之后，下一步可能涉及跟踪对象。SDK中有几个内置的参考跟踪器，从高性能到高精度。使用Gst nvtracker插件执行对象跟踪。

为了创建可视化工件，如边界框、分割掩码和标签，有一个名为Gst-nvdosd的可视化插件。

最后，为了输出结果，DeepStream提供了各种选项：在屏幕上用边界框渲染输出，将输出保存到本地磁盘，通过RTSP流式传输或仅将元数据发送到云。
为了将元数据发送到云，DeepStream使用Gst-nvmsgconv和Gst-nvmsgbroker插件。
Gst-nvmsgconv将元数据转换为模式有效载荷，Gst-nvmsgbroker建立到云的连接并发送遥测数据。
有几个内置的代理协议，如Kafka、MQTT、AMQP和Azure IoT。可以创建自定义代理适配器。

## DeepStream reference app

首先，开发人员可以使用提供的参考应用程序。还包括这些应用程序的源代码。端到端应用程序称为深度流应用程序。
此应用程序是完全可配置的-它允许用户配置任何类型和数量的源。用户还可以选择要运行推理的网络类型。
它预装了一个用于进行对象检测的推理插件，该插件由用于进行图像分类的推理插件级联而成。
有一个配置跟踪器的选项。对于输出，用户可以在屏幕上渲染、保存输出文件或通过RTSP流式传输视频之间进行选择。
![image](https://github.com/popcornGit/deepstream_tran/assets/48575896/f65d830b-b002-4e79-b62d-cb8d32e62b45)

这是一个很好的参考应用程序，可以开始学习DeepStream的功能。
该应用程序在DeepStream参考应用程序-DeepStream应用程序一章中有更详细的介绍。
此应用程序的源代码位于/opt/nvidia/deepstream/deepstream-6.3/sources/apps/sample_apps/deepstream-app中。
该应用程序适用于所有人工智能模型，并在个人自述文件中提供详细说明。性能基准测试也使用此应用程序运行。

## Getting started with building apps
对于希望构建自定义应用程序的开发人员来说，深度流应用程序在开始开发时可能会有点势不可挡。
SDK附带了几个简单的应用程序，开发人员可以在其中学习DeepStream的基本概念，构建一个简单的管道，然后继续构建更复杂的应用程序。
![image](https://github.com/popcornGit/deepstream_tran/assets/48575896/b74fae76-fb21-4e3d-b85d-45ef0dc42ec0)

开发人员可以从deepstream-test1开始，它几乎就像一个deepstream hello世界。
在这个应用程序中，开发人员将学习如何使用各种DeepStream插件构建GStreamer管道。
他们将从文件中获取视频，解码、批处理，然后进行对象检测，最后在屏幕上渲染方框。
深度流测试2从测试1开始，并将辅助网络级联到主网络。深度流测试3展示了如何添加多个视频源，最后测试4将展示如何使用消息代理插件提供物联网服务。
这4个入门应用程序既有原生C/C++，也有Python。
要在DeepStream中阅读更多关于这些应用程序和其他示例应用程序的信息，请参阅C/C++示例应用程序源详细信息和Python示例应用程序和绑定源详细信息。

DeepStream应用程序可以在不使用Graph Composer进行编码的情况下创建。有关详细信息，请参阅图形生成器简介。

## DeepStream in Python
Python易于使用，在创建人工智能模型时被数据科学家和深度学习专家广泛采用。
NVIDIA引入了Python绑定，帮助您使用Python构建高性能人工智能应用程序。DeepStream管道可以使用GStreamer框架的Python绑定GstPython构建。
![image](https://github.com/popcornGit/deepstream_tran/assets/48575896/98c54498-8714-45e1-af80-9729500f819d)

DeepStream Python应用程序使用Gst-Python API操作来构建管道，并使用探测函数来访问管道中各个点的数据。
数据类型都是原生C，需要通过PyBindings或NumPy的填充层才能从Python应用程序访问它们。
张量数据是经过推理后得出的原始张量输出。如果您试图检测对象，则需要通过解析和聚类算法对该张量数据进行后处理，以在检测到的对象周围创建边界框。
要开始使用Python，请参阅本指南中的Python示例应用程序和绑定源详细信息以及DeepStreamPython API指南中的“DeepStreamPython”。
