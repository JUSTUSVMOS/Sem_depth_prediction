# 若要一次模擬複數張SEM只要執行此檔案即可
#!/bin/bash
set -e

# 導出路徑 PATH=$PATH:/path/to/your/build/bin
export PATH=$PATH:/mnt/d/elsepa/nebula/build/bin

# 運行 python sem-pri.py
python sem-pri.py

# 循環從 0 到 999  (根據要轉換的.tri檔 數量做調整)
for i in {0..999}
do
    # 格式化數字為兩位數（如 0000, 0001, 0002, ... 0099）
    formatted_i=$(printf "%04d" $i)
    
    # 檢查 .tri 文件是否存在
    if [ ! -f triangles_${formatted_i}.tri ]; then
        echo "File triangles_${formatted_i}.tri does not exist!"
        exit 1
    fi
    
    echo "Processing triangles_${formatted_i}.tri ..."
    
    # 運行 nebula_gpu 命令將triangles_0000.tri檔輸出一個output_0000.det檔供分析腳本分析(這邊名稱可自行更改)
    nebula_gpu triangles_${formatted_i}.tri sem.pri silicon.mat pmma.mat > output_${formatted_i}.det
    
    # 運行 python 分析腳本，傳遞 formatted_i 作為參數
    python sem-analysis.py output_${formatted_i}.det $formatted_i
done
