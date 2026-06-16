# VegCover - 航拍照片拼接与植被分析工具 设计文档

**日期**: 2026-06-16
**状态**: 已确认

## 概述

VegCover 是一个桌面端航拍照片分析工具，将多张 RGB 航拍照片自动拼接为全景图，识别植被种类并统计覆盖率。采用线性 Pipeline 架构，Gradio Web UI 交互。

## 需求

- **界面**: 桌面 GUI (PyQt6)
- **拼接**: 自动特征匹配（OpenCV Stitcher，ORB 特征）
- **植被识别**: YOLO 预训练模型起步 + 支持加载自定义 .pt 模型
- **植被指数**: RGB 照片使用 ExG（过量绿指数）和 VARI（可见光抗大气植被指数）
- **融合策略**: YOLO 提供语义类别，植被指数补充漏检区域
- **输出**: 拼接全景图+植被标注、覆盖率统计报表、可视化图表（饼图/柱状图/热力图）、导出文件（CSV/JSON/PNG）

## 架构

### 整体数据流

```
用户上传 → [Stitcher] → [YOLO Detector] → [VegIndex Calc] → [Stats] → [桌面 GUI]
```

1. 用户上传多张 RGB 航拍照片
2. `stitch()` — OpenCV Stitcher 特征匹配拼接，输出全景图
3. `detect()` — YOLO 模型检测植被区域，输出边界框+类别
4. `vegindex()` — 计算 ExG/VARI 植被指数，生成植被概率图
5. `stats()` — 融合 YOLO 检测结果 + 植被指数图，计算各类别面积和覆盖率
6. `export()` — 生成标注图、统计报表、可视化图表、CSV/JSON 数据文件

### 融合策略

- YOLO 框内像素：赋予类别标签
- YOLO 框外 + 指数为植被：标为"未分类植被"
- 框外 + 指数为非植被：标为"非植被"

## 模块设计

### 照片拼接模块 (stitcher.py)

- 预处理：统一分辨率、可选直方图均衡化
- 特征提取+匹配：ORB 关键点，BFMatcher 匹配
- 单应性矩阵：RANSAC 估计变换矩阵
- 图像融合：MultiBandBlender 多频段融合，消除拼接缝

**异常处理**:
- 照片无重叠 → 提示用户，退回单张模式
- 特征点不足 → 降低匹配阈值重试
- 拼接结果严重变形 → 警告用户，展示最佳单张

**可配置参数**: 特征匹配阈值、融合方式、是否直方图均衡化

### YOLO 植被检测模块 (detector.py)

- 默认加载 YOLOv8n 预训练模型
- 支持指定自定义 .pt 模型文件路径
- COCO 预训练类别映射：plant/tree 等
- 自定义模型可定义更细类别（乔木、灌木、草本、农作物、裸地等）

**可配置参数**: 置信度阈值、IOU 阈值、自定义模型路径

### 可见光植被指数模块 (vegindex.py)

- **ExG（过量绿指数）**: `ExG = 2G - R - B`，强调绿色植被，对草地/作物效果好
- **VARI（可见光抗大气植被指数）**: `VARI = (G-R) / (G+R-B)`，抗大气干扰，对乔木/灌木效果更好
- Otsu 自动阈值二值化，生成像素级植被概率图
- 融合两种指数生成最终植被掩膜

### 覆盖率统计模块 (stats.py)

- 逐类别统计：各植被类别像素数 / 总像素数 = 覆盖率
- 面积估算：用户可输入 GSD（地面采样距离，m/pixel），换算真实面积（m²/公顷）
- 不输入 GSD 则只报告像素占比
- 统计总植被覆盖率和非植被覆盖率

### 导出模块 (exporter.py)

**可视化图表**:

| 图表类型 | 内容 | 实现方式 |
|---------|------|---------|
| 标注全景图 | 全景图 + YOLO 边界框 + 植被掩膜色彩 | OpenCV 绘制 |
| 饼图 | 各类别覆盖率占比 | Matplotlib |
| 柱状图 | 各类别像素数/面积对比 | Matplotlib |
| 热力图 | 植被指数空间分布 | Matplotlib + ExG/VARI |

**导出文件**:

| 格式 | 内容 |
|------|------|
| PNG/JPG | 标注全景图、热力图 |
| CSV | 逐类别统计表（类别、像素数、覆盖率、面积） |
| JSON | 完整分析结果（含检测框坐标、统计数值、参数配置） |

所有导出文件打包到输出目录，UI 上提供单独下载按钮。

## UI 设计 (PyQt6 桌面应用)

单窗口桌面应用，三个区域：

1. **输入区**: 选择多张航拍照片，预览缩略图
2. **参数配置区**: 拼接参数、检测参数、植被指数参数、导出参数
3. **结果展示区**: 四个 Tab 切换
   - 拼接结果：全景图预览
   - 植被标注：全景图 + YOLO 框 + 掩膜叠加
   - 统计图表：饼图 + 柱状图 + 热力图
   - 数据导出：报表预览 + 导出按钮

**交互**: 分析过程后台线程运行，进度条显示当前步骤，大图支持缩放。

## 项目结构

```
VegCover/
├── pyproject.toml
├── main.py                  # 入口：启动 PyQt6 桌面应用
├── src/
│   └── vegcover/
│       ├── __init__.py
│       ├── pipeline.py      # Pipeline 控制器
│       ├── stitcher.py      # 照片拼接模块
│       ├── detector.py      # YOLO 植被检测模块
│       ├── vegindex.py      # 可见光植被指数计算模块
│       ├── stats.py         # 覆盖率统计模块
│       ├── exporter.py      # 报表/图表/文件导出模块
│       └── models.py        # 数据模型定义
├── output/                  # 默认导出目录（gitignore）
└── tests/
    └── test_pipeline.py     # 核心流程单元测试
```

## 依赖

| 包 | 用途 |
|---|------|
| `PyQt6` | 桌面 GUI 框架 |
| `opencv-python` | 图像拼接、特征匹配、绘制 |
| `ultralytics` | YOLOv8/v11 模型推理 |
| `numpy` | 数值计算 |
| `matplotlib` | 统计图表 |
| `pillow` | 图像读写辅助 |

## 数据模型 (models.py)

```python
@dataclass
class Box:
    x1: int; y1: int; x2: int; y2: int
    class_name: str
    confidence: float

@dataclass
class StitchResult:
    panorama: np.ndarray
    success: bool
    error_msg: str | None
    transform_matrices: list

@dataclass
class DetectResult:
    boxes: list[Box]
    classes: list[str]
    model_name: str

@dataclass
class VegIndexMap:
    exg_map: np.ndarray
    vari_map: np.ndarray
    veg_mask: np.ndarray
    veg_ratio: float

@dataclass
class CategoryStat:
    name: str
    pixels: int
    ratio: float
    area_m2: float | None

@dataclass
class CoverageStat:
    total_pixels: int
    categories: list[CategoryStat]
    total_veg_ratio: float
    non_veg_ratio: float
    gsd: float | None

@dataclass
class AnalysisResult:
    stitch: StitchResult
    detect: DetectResult
    veg_index: VegIndexMap
    coverage: CoverageStat
```
