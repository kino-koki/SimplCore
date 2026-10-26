# -*- coding: utf-8 -*-
"""纯标准库生成 .docx（不依赖 python-docx）。
将 report_part1.md + report_appendix.md 中的结构化 Markdown 解析为 docx。
支持：#/##/### 标题、普通段落、> 引用、表格（| 分隔）、无序列表（- / *）、有序列表、代码块。
"""
import re
import zipfile
import os
from xml.sax.saxutils import escape

PART1 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part1.md"
PART2 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part2.md"
PART3 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part3.md"
PART4 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part4.md"
PART5 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part5.md"
PART6 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part6.md"
PART7 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part7.md"
PART8 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part8.md"
PART9 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part9.md"
PART10 = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_part10.md"
FIXUP = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_fixup.md"
APPENDIX = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\report_appendix.md"
OUT = r"G:\青小团多模态语言大模型 Qingxiaotuan LLM v1.0\SimplCore架构完整性深度评估报告.docx"

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')

def para(text, style=None, bold=False, size=None, color=None, align=None, space_after=120):
    rpr = ""
    if bold or size or color:
        rpr = "<w:rPr>"
        if bold:
            rpr += "<w:b/>"
        if color:
            rpr += f'<w:color w:val="{color}"/>'
        if size:
            rpr += f'<w:sz w:val="{size*2}"/>'
        rpr += "</w:rPr>"
    ppr = "<w:pPr>"
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    ppr += f'<w:spacing w:after="{space_after}"/>'
    ppr += "</w:pPr>"
    runs = ""
    # 支持行内 **bold**
    parts = re.split(r'(\*\*[^*]+\*\*)', text)
    for p in parts:
        if p.startswith("**") and p.endswith("**"):
            runs += f'<w:r>{rpr}<w:t xml:space="preserve">{escape(p[2:-2])}</w:t></w:r>'
        else:
            if p:
                runs += f'<w:r>{rpr}<w:t xml:space="preserve">{escape(p)}</w:t></w:r>'
    return f"<w:p>{ppr}{runs}</w:p>"

def heading(text, level):
    sizes = {1: 22, 2: 18, 3: 15, 4: 13}
    colors = {1: "1F3864", 2: "2E5496", 3: "2E5496", 4: "404040"}
    return para(text, bold=True, size=sizes.get(level, 13), color=colors.get(level, "000000"), space_after=200)

def table(rows):
    # rows: list of list[str]; first row = header
    out = ['<w:tbl>']
    out.append('<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
               '<w:tblBorders>'
               '<w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '<w:left w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '<w:right w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
               '</w:tblBorders></w:tblPr>')
    for ri, row in enumerate(rows):
        out.append('<w:tr>')
        for cell in row:
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9E2F3"/>' if ri == 0 else ''
            bold = '<w:b/>' if ri == 0 else ''
            out.append(
                '<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/>' + shade +
                '<w:tcBorders><w:top w:val="single" w:sz="4" w:color="999999"/>'
                '<w:left w:val="single" w:sz="4" w:color="999999"/>'
                '<w:bottom w:val="single" w:sz="4" w:color="999999"/>'
                '<w:right w:val="single" w:sz="4" w:color="999999"/></w:tcBorders>'
                '</w:tcPr>'
                f'<w:p><w:pPr><w:spacing w:after="40"/></w:pPr>'
                f'<w:r><w:rPr>{bold}</w:rPr><w:t xml:space="preserve">{escape(cell)}</w:t></w:r></w:p>'
                '</w:tc>')
        out.append('</w:tr>')
    out.append('</w:tbl>')
    out.append(para("", space_after=80))
    return "".join(out)

def parse_md(text):
    lines = text.split("\n")
    body = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # 代码块
        if line.strip().startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            ctext = "\n".join(code)
            body.append(
                '<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>'
                '<w:spacing w:after="120"/></w:pPr>'
                f'<w:r><w:rPr><w:sz w:val="18"/><w:color w:val="333333"/></w:rPr>'
                f'<w:t xml:space="preserve">{escape(ctext)}</w:t></w:r></w:p>')
            continue
        # 表格
        if line.strip().startswith("|") and i + 1 < n and re.match(r'^\s*\|[-\s|:]+\|\s*$', lines[i+1]):
            rows = []
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body.append(table(rows))
            continue
        # 标题
        m = re.match(r'^(#{1,4})\s+(.*)$', line)
        if m:
            lvl = len(m.group(1))
            body.append(heading(m.group(2).strip(), lvl))
            i += 1
            continue
        # 引用
        if line.strip().startswith(">"):
            q = line.strip()[1:].strip()
            body.append(
                '<w:p><w:pPr><w:ind w:left="360"/><w:shd w:val="clear" w:color="auto" w:fill="EAF0FB"/>'
                '<w:spacing w:after="120"/></w:pPr>'
                f'<w:r><w:rPr><w:color w:val="1F3864"/></w:rPr>'
                f'<w:t xml:space="preserve">{escape(q)}</w:t></w:r></w:p>')
            i += 1
            continue
        # 列表
        if re.match(r'^\s*[-*]\s+', line):
            item = re.sub(r'^\s*[-*]\s+', '', line)
            body.append(
                '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
                '<w:spacing w:after="60"/></w:pPr>'
                f'<w:r><w:t xml:space="preserve">{escape(item)}</w:t></w:r></w:p>')
            i += 1
            continue
        if re.match(r'^\s*\d+\.\s+', line):
            item = re.sub(r'^\s*\d+\.\s+', '', line)
            body.append(
                '<w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
                '<w:spacing w:after="60"/></w:pPr>'
                f'<w:r><w:t xml:space="preserve">{escape(item)}</w:t></w:r></w:p>')
            i += 1
            continue
        # 空行
        if line.strip() == "":
            i += 1
            continue
        # 普通段落
        body.append(para(line.strip()))
        i += 1
    return "".join(body)

def build():
    with open(PART1, encoding="utf-8") as f:
        t1 = f.read()
    with open(PART2, encoding="utf-8") as f:
        t2 = f.read()
    with open(PART3, encoding="utf-8") as f:
        t3 = f.read()
    with open(PART4, encoding="utf-8") as f:
        t4 = f.read()
    with open(PART5, encoding="utf-8") as f:
        t5 = f.read()
    with open(PART6, encoding="utf-8") as f:
        t6 = f.read()
    with open(PART7, encoding="utf-8") as f:
        t7 = f.read()
    with open(PART8, encoding="utf-8") as f:
        t8 = f.read()
    with open(PART9, encoding="utf-8") as f:
        t9 = f.read()
    with open(PART10, encoding="utf-8") as f:
        t10 = f.read()
    with open(FIXUP, encoding="utf-8") as f:
        t11 = f.read()
    with open(APPENDIX, encoding="utf-8") as f:
        t12 = f.read()
    content = parse_md(t1 + "\n\n" + t2 + "\n\n" + t3 + "\n\n" + t4 + "\n\n" + t5 + "\n\n" + t6 + "\n\n" + t7 + "\n\n" + t8 + "\n\n" + t9 + "\n\n" + t10 + "\n\n" + t11 + "\n\n" + t12)
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document {NS}><w:body>{content}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/></w:sectPr>'
        '</w:body></w:document>')

    numbering_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:numbering {NS}>'
        '<w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0">'
        '<w:numFmt w:val="bullet"/><w:lvlText w:val="\u2022"/>'
        '<w:lvlJc w:val="left"/></w:lvl></w:abstractNum>'
        '<w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0">'
        '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
        '<w:lvlJc w:val="left"/></w:lvl></w:abstractNum>'
        '<w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
        '<w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>'
        '</w:numbering>')

    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:styles {NS}>'
        '<w:docDefaults><w:rPrDefault><w:rPr>'
        '<w:rFonts w:ascii="Calibri" w:eastAsia="\u5b8b\u4f53" w:hAnsi="Calibri"/>'
        '<w:sz w:val="21"/></w:rPr></w:rPrDefault></w:docDefaults>'
        '<w:style w:type="paragraph" w:styleId="TableGrid"><w:name w:val="Table Grid"/>'
        '<w:tblPr><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:color="999999"/>'
        '<w:left w:val="single" w:sz="4" w:color="999999"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="999999"/>'
        '<w:right w:val="single" w:sz="4" w:color="999999"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="999999"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="999999"/>'
        '</w:tblBorders></w:tblPr></w:style>'
        '</w:styles>')

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
        '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
        '</Types>')

    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>')

    doc_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
        '</Relationships>')

    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:title>SimplCore 架构完整性深度评估报告</dc:title>'
        '<dc:creator>WorkBuddy 架构评估</dc:creator>'
        '<cp:lastModifiedBy>WorkBuddy</cp:lastModifiedBy>'
        '<dcterms:created xsi:type="dcterms:W3CDTF">2026-08-22T22:01:00+08:00</dcterms:created>'
        '<dcterms:modified xsi:type="dcterms:W3CDTF">2026-08-22T22:01:00+08:00</dcterms:modified>'
        '</cp:coreProperties>')

    app = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        '<Application>WorkBuddy</Application><Company>\u9752\u5c0f\u56e2</Company></Properties>')

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document_xml)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles_xml)
        z.writestr("word/numbering.xml", numbering_xml)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
    size = os.path.getsize(OUT)
    print(f"WROTE {OUT}  ({size} bytes)")

if __name__ == "__main__":
    build()
