import matplotlib.font_manager as fm
available_fonts = [f.name for f in fm.fontManager.ttflist]
print("Available Fonts:", available_fonts)