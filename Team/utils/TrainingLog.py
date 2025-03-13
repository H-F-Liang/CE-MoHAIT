def save_log(log_file, accuracy):
    # 打开指定的日志文件并记录数据
    with open(log_file, 'a') as f:
        log_string = f"{accuracy}\n"
        f.write(log_string)

    print(f"Log saved to {log_file}")

