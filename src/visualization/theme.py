import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

# ======================
# 核心视觉元素定义
# ======================
class CNSStyle:
    """Nature/Cell级期刊图表样式规范"""
    
    # --- 主色系 ---
    PRIMARY_PALETTE = {
        'blue': '#0072B2',    # 科学蓝 (Nature主色)
        'orange': '#D55E00',  # 活力橙 (Cell强调色)
        'green': '#009E73',   # 森林绿 (色盲友好)
        'red': '#CC79A7',     # 宝石红 (CNS通用)
        'gray': '#4D4D4D',    # 石墨灰 (优化对比度)
        'black': '#2B2B2B'    # 深空黑
    }
    
    # --- 辅助色系 ---
    SECONDARY_PALETTE = {
        'light_blue': '#56B4E9',
        'sand': '#E69F00',
        'mint': '#40B0A6',
        'lavender': '#9E73C7'
    }
    
    # --- 专业色阶 ---
    @classmethod
    def create_cmap(cls, name='cns_thermal', gradient_type='cns_thermal'): # Modified to take gradient_type
        """创建符合出版要求的渐变色阶"""
        gradients = {
            'cns_thermal': [cls.PRIMARY_PALETTE['blue'], 
                           cls.SECONDARY_PALETTE['light_blue'],
                           cls.PRIMARY_PALETTE['orange']],
            'cns_diverging': [cls.PRIMARY_PALETTE['green'],
                              '#FFFFFF', 
                              cls.PRIMARY_PALETTE['red']]
        }
        return LinearSegmentedColormap.from_list(name, gradients[gradient_type]) # Use gradient_type


# ======================
# 样式配置引擎
# ======================
def configure_cns_style(debug=False):
    """
    应用CNS级期刊图表样式配置
    
    Parameters:
    debug (bool): 启用调试模式显示配置参数
    """
    
    plt.style.use('seaborn-v0_8-whitegrid') # Changed from 'seaborn-whitegrid' to a versioned one

    # --- 字体配置 ---
    font_config = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica Neue', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 18,
        'mathtext.fontset': 'stixsans',  # 数学符号专用字体
        'mathtext.default': 'regular'
    }
    
    # --- 几何元素配置 ---
    element_config = {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.2,
        'axes.edgecolor': CNSStyle.PRIMARY_PALETTE['black'],
        'axes.labelcolor': CNSStyle.PRIMARY_PALETTE['black'],
        'xtick.color': CNSStyle.PRIMARY_PALETTE['black'],
        'ytick.color': CNSStyle.PRIMARY_PALETTE['black'],
        'grid.color': '#EAEAEA',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'lines.markersize': 8.5,
        'patch.linewidth': 1.2,
        'hatch.linewidth': 0.8
    }
    
    # --- 颜色循环器 ---
    color_cycle = cycler(color=[
        CNSStyle.PRIMARY_PALETTE['blue'],
        CNSStyle.PRIMARY_PALETTE['orange'],
        CNSStyle.PRIMARY_PALETTE['green'],
        CNSStyle.PRIMARY_PALETTE['red'],
        CNSStyle.SECONDARY_PALETTE['light_blue']
    ])
    
    # 合并配置
    plt.rcParams.update({**font_config, **element_config})
    plt.rc('axes', prop_cycle=color_cycle)
    
    # The register_cmap calls are handled outside this function now.
    
    if debug:
        print("[DEBUG] Current RC Params:\n", 
              {k:v for k,v in plt.rcParams.items() if 'size' in k or 'color' in k})

# ======================
# 高级功能扩展
# ======================
class CNSTools:
    """CNS级图表增强工具集"""
    
    @staticmethod
    def add_style_marker(ax=None, position='lower right'):
        """添加蒙德里安风格定位标记"""
        ax = ax or plt.gca()
        positions = {
            'lower right': (0.78, 0.05),
            'upper left': (0.12, 0.88)
        }
        x, y = positions.get(position, (0.78, 0.05))
        
        marker = Rectangle((x, y), 0.2, 0.1, 
                           transform=ax.transAxes,
                           fill=False,
                           edgecolor=CNSStyle.PRIMARY_PALETTE['black'],
                           linewidth=1.5,
                           linestyle='--')
        ax.add_patch(marker)
    
    @staticmethod
    def save_for_publication(filename, dpi=600, **kwargs):
        """期刊级图表保存预设"""
        save_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': False,
            'metadata': {
                'Creator': 'Matplotlib v3.x',
                'Producer': 'PDFlib',
                'CreationDate': None  # 符合双盲评审要求
            }
        }
        save_kwargs.update(kwargs)
        plt.savefig(filename, **save_kwargs)

# ======================
# 上下文管理器
# ======================
class cns_style_context:
    """样式上下文管理器，支持临时样式切换"""
    
    def __init__(self, style='cns', palette='default'):
        self._original_rc = plt.rcParams.copy()
        self.style = style
        self.palette = palette
        
    def __enter__(self):
        if self.style == 'cns':
            configure_cns_style()
        elif self.style == 'grayscale':
            self._apply_grayscale()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.rcParams.update(self._original_rc)
        
    def _apply_grayscale(self):
        """灰度出版模式"""
        grayscale_cycle = cycler(color=['#404040', '#808080', '#A0A0A0', '#C0C0C0'])
        plt.rc('axes', prop_cycle=grayscale_cycle)
        plt.rcParams.update({
            'axes.edgecolor': '#000000',
            'grid.color': '#CCCCCC'
        })

# ======================
# 初始化配置 (Executes ONCE when module is imported)
# ======================
# This block executes when theme.py is imported or run directly.
# We ensure colormaps are registered using matplotlib.colormaps.register
# which is supported in newer matplotlib versions (3.4+).
# For older versions, custom colormaps can be passed directly to plotting functions
# or explicitly registered with the older `plt.cm.register_cmap` if that's available.

# Check if matplotlib.colormaps.register is available (Matplotlib 3.4+)
if hasattr(plt.colormaps, 'register'):
    if 'cns_thermal' not in plt.colormaps():
        plt.colormaps.register(cmap=CNSStyle.create_cmap(name='cns_thermal', gradient_type='cns_thermal'))
    if 'cns_diverging' not in plt.colormaps():
        plt.colormaps.register(cmap=CNSStyle.create_cmap(name='cns_diverging', gradient_type='cns_diverging'))
elif hasattr(plt.cm, 'register_cmap'): # Fallback for older versions of Matplotlib
    if 'cns_thermal' not in plt.colormaps(): # Still check colormaps() for existing
        plt.cm.register_cmap(cmap=CNSStyle.create_cmap(name='cns_thermal', gradient_type='cns_thermal'))
    if 'cns_diverging' not in plt.colormaps():
        plt.cm.register_cmap(cmap=CNSStyle.create_cmap(name='cns_diverging', gradient_type='cns_diverging'))
else:
    print("Warning: Neither plt.colormaps.register nor plt.cm.register_cmap found. Custom colormaps may not be available.")

# Apply standard theme settings (non-debug) automatically upon import.
configure_cns_style(debug=False)