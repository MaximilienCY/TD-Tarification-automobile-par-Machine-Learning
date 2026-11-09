"""Repair point-label placement omitted by the Artifact Tool line-chart exporter.

All data and charts remain native/editable. This only places the existing eight
public-score labels above/below their points so neighboring values do not overlap.
"""
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

path = Path(sys.argv[1])
ns={"c":"http://schemas.openxmlformats.org/drawingml/2006/chart"}
ET.register_namespace("c",ns["c"])
ET.register_namespace("a","http://schemas.openxmlformats.org/drawingml/2006/main")
ET.register_namespace("r","http://schemas.openxmlformats.org/officeDocument/2006/relationships")
values=[]
with zipfile.ZipFile(path) as z:
    for item in z.infolist():
        data=z.read(item.filename)
        if "/charts/" in item.filename and item.filename.endswith(".xml"):
            root=ET.fromstring(data)
            for axis in root.findall('.//c:catAx',ns):
                for name in ['tickLblSkip','tickMarkSkip']:
                    tag=axis.find('c:'+name,ns)
                    if tag is None:
                        tag=ET.Element('{'+ns['c']+'}'+name)
                        later=[v for v in list(axis) if v.tag.split('}')[-1] in ['noMultiLvlLbl','extLst']]
                        axis.insert(list(axis).index(later[0]) if later else len(axis),tag)
                    tag.set('val','1')
            chart=root.find(".//c:lineChart",ns)
            if chart is not None:
                labels=chart.find("c:dLbls",ns)
                if labels is None:
                    labels=ET.SubElement(chart,"{"+ns["c"]+"}dLbls")
                for old in labels.findall("c:dLbl",ns):labels.remove(old)
                for idx,pos in enumerate(["t","b","t","t","b","t","t","t"]):
                    label=ET.Element("{"+ns["c"]+"}dLbl")
                    ET.SubElement(label,"{"+ns["c"]+"}idx",{"val":str(idx)})
                    ET.SubElement(label,"{"+ns["c"]+"}dLblPos",{"val":pos})
                    for flag,val in [('showLegendKey',0),('showVal',1),('showCatName',0),('showSerName',0),('showPercent',0),('showBubbleSize',0)]:
                        ET.SubElement(label,'{'+ns['c']+'}'+flag,{'val':str(val)})
                    labels.insert(idx,label)
            data=ET.tostring(root,encoding="utf-8",xml_declaration=True)
        values.append((item,data))
temp=path.with_suffix(".labels.tmp")
with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for item,data in values:z.writestr(item,data)
temp.replace(path)
print("Native line-chart labels positioned, all source values preserved.")
