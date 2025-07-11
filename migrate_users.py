import os
from database import add_user, get_user_by_name, create_users_table
from config import config # 获取数据目录

def migrate_existing_users():
    print("开始将现有用户数据迁移到MySQL...")
    create_users_table() # 迁移前确保表存在

    data_root_dir = config.get('data_dir', 'data/') # 从配置中获取数据目录

    # 如果data_dir在配置中不是绝对路径，则将其调整为相对于项目根目录的路径
    # 假设config中的data_dir是相对于项目根目录的
    if not os.path.isabs(data_root_dir):
        data_root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_root_dir)

    if not os.path.exists(data_root_dir):
        print(f"错误：未找到数据目录 '{data_root_dir}'。没有用户可迁移。")
        return

    migrated_count = 0
    skipped_count = 0

    for user_name in os.listdir(data_root_dir):
        user_folder_path = os.path.join(data_root_dir, user_name)
        if os.path.isdir(user_folder_path):
            # 检查用户是否已存在于数据库中
            if get_user_by_name(user_name):
                print(f"用户 '{user_name}' 已存在于数据库中。跳过。")
                skipped_count += 1
                continue

            # 查找用户文件夹中的第一张图片
            # 首先尝试'1.jpg'（通常用于相机采集）
            first_image_found = False
            photo_path = None

            potential_first_image = os.path.join(user_folder_path, "1.jpg")
            if os.path.exists(potential_first_image):
                photo_path = potential_first_image
                first_image_found = True
            else:
                # 查找任何.jpg文件
                for filename in os.listdir(user_folder_path):
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        photo_path = os.path.join(user_folder_path, filename)
                        first_image_found = True
                        break

            if first_image_found:
                user_id = add_user(user_name, photo_path)
                if user_id:
                    print(f"已迁移用户 '{user_name}'，ID: {user_id}")
                    migrated_count += 1
                else:
                    print(f"迁移用户 '{user_name}' 失败。")
            else:
                print(f"未找到用户 '{user_name}' 的图片。跳过迁移。")
                skipped_count += 1
    
    print(f"\n迁移完成。成功迁移 {migrated_count} 个用户。跳过 {skipped_count} 个用户。")

if __name__ == "__main__":
    migrate_existing_users()