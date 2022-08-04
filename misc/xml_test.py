import xml.etree.ElementTree as ET

import xmltodict

# tree = ET.parse('./config/franka_ex.xml')



# root = tree.getroot()
# print(root.tag)


# for child in root:
#     print(child.tag, child.attrib,)


with open("./config/franka_ex.xml") as f:
    hoge = f.read()


dict_xml = xmltodict.parse(hoge)
print(dict_xml["simulation_param"])






# xml2 = """
#     <student>
#         <id name="matsuta" age="10"/>
#     </student>
# """


# dic2 = xmltodict.parse(xml2)
# print(dic2["student"]["id"]["@name"])