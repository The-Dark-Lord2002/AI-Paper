"""Sets the complex-script (Persian) font of the deck's theme to Vazirmatn, so text without an explicit
font also renders in Vazirmatn in PowerPoint."""
import re, shutil, sys, zipfile
src = sys.argv[1]; tmp = src + ".tmp"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename.startswith("ppt/theme/") and item.filename.endswith(".xml"):
            xml = data.decode("utf-8")
            xml = xml.replace('<a:cs typeface=""/>', '<a:cs typeface="Vazirmatn"/>')
            xml = re.sub(r'<a:font script="Arab" typeface="[^"]*"/>', '<a:font script="Arab" typeface="Vazirmatn"/>', xml)
            data = xml.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, src)
print("theme fonts set:", src)
