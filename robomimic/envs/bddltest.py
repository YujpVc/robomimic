from bddl import parse_bddl_file
conditions = parse_bddl_file("/home/yujp/MimicPlay/mimicplay/scripts/bddl_files/mybddl.bddl")
print(conditions)  # 查看解析后的逻辑树