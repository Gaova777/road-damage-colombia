from road_damage_colombia.adapters.rdd2022 import read_voc, to_yolo

XML = """<annotation><size><width>600</width><height>600</height></size>
<object><name>D40</name><bndbox><xmin>100</xmin><ymin>300</ymin><xmax>200</xmax><ymax>400</ymax></bndbox></object>
<object><name>Repair</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object>
<object><name>D00</name><bndbox><xmin>590</xmin><ymin>10</ymin><xmax>700</xmax><ymax>20</ymax></bndbox></object>
</annotation>"""


def test_only_damage_classes_and_boxes_clipped_to_the_image(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(XML)
    w, h, boxes = read_voc(p)
    assert [b[0] for b in boxes] == ["D40", "D00"]
    lines = to_yolo(w, h, boxes)
    assert lines[0] == "3 0.250000 0.583333 0.166667 0.166667"
    assert lines[1].startswith("0 0.991667")  # x2=700 clipped to 600
