def preprocess_data(filename_in='../data/second_processed.csv', filename_out='', model_type='BIMODAL', starting_point='random',
                    invalid=True, duplicates=True, salts=True, stereochem=True, canonicalize=True, min_len=14,
                    max_len=74, augmentation=1):

    try:
        from preprocessor import Preprocessor
        print("Preprocessor module imported successfully.")
    except ImportError as e:
        print(f"Error importing Preprocessor: {e}")
        return

    # 读取原始文件的所有列
    import pandas as pd
    original_df = pd.read_csv(filename_in, header=None)
    other_columns = original_df.iloc[:, 1:]  # 保存其他列数据
    first_column = original_df.iloc[:, 0].to_frame()  # 提取第一列

    # 将第一列临时保存供Preprocessor处理
    temp_file = "../data/temp_first_col.csv"
    first_column.to_csv(temp_file, index=False, header=False)

    # 初始化Preprocessor处理第一列
    p = Preprocessor(temp_file.replace('.csv', ''))  # 注意Preprocessor会自动加.csv
    print(f'Pre-processing first column of "{filename_in}" started.')

    # 准备模型类型名称
    dataname = filename_in.split('/')[-1]
    if model_type == "ForwardRNN":
        name = model_type
    else:
        name = f"{model_type}_{starting_point}"
        if augmentation > 1 and starting_point == 'fixed':
            augmentation = 1

    # 执行预处理
    p.preprocess(name, aug=augmentation, length=max_len)

    # 获取处理后的第一列数据
    processed_first_col = p.get_data()

    # 合并处理后的第一列和原始其他列
    final_df = pd.DataFrame(processed_first_col)
    if not other_columns.empty:
        final_df = pd.concat([final_df, other_columns], axis=1)

    # 设置输出路径
    if not filename_out:
        filename_out = f'../data/final_{name}.csv'

    # 保存最终数据
    final_df.to_csv(filename_out, index=False, header=False)
    print(f'Processed data saved to {filename_out}')

    # 清理临时文件
    import os
    os.remove(temp_file)

if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"An error occurred: {e}")