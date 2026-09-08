import csv
import os

def test_csv_write_read():
    """测试CSV文件读写"""
    # 写入
    with open(".lab/test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["姓名", "年龄", "城市"])
        writer.writerow(["张三", 25, "北京"])
        writer.writerow(["李四", 30, "上海"])

    # 读取
    with open(".lab/test.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)

def test_txt_write_read():
    """测试TXT文件读写"""
    # 写入
    with open(".lab/test.txt", "w", encoding="utf-8") as f:
        f.write("这是第一行文本\n")
        f.write("这是第二行文本\n")

    # 读取
    with open(".lab/test.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

if __name__ == "__main__":
    test_csv_write_read()
    test_txt_write_read()