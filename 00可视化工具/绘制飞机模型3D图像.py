import matplotlib.pyplot as plt
from datetime import datetime
import os
import matplotlib.animation as animation
from PIL import Image

home_directory = os.path.expanduser('~')
desktop_path = os.path.join(home_directory, 'Desktop')
print(desktop_path)
#desktop_path='C:\\000 onedrive 备份\\OneDrive\\桌面'
# 定义正方体的顶点和面
cube_vertices = [
    [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
]
cube_faces = [
    [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]
]

# 定义四棱锥的顶点和面
pyramid1_vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [3, 0, -1]]
pyramid1_faces = [[0, 1, 3, 2], [0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]

pyramid2_vertices = [[1, 1, 1], [1, 1, -1], [-1, 1, 1], [-1, 1, -1], [-1, 3, -1]]
pyramid2_faces = [[0, 1, 3, 2], [0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]

pyramid3_vertices = [[1, -1, 1], [1, -1, -1], [-1, -1, 1], [-1, -1, -1], [-1, -3, -1]]
pyramid3_faces = [[0, 1, 3, 2], [0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]

pyramid4_vertices = [[1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1], [-1, 0, 3]]
pyramid4_faces = [[0, 1, 3, 2], [0, 1, 4], [0, 2, 4], [1, 3, 4], [2, 3, 4]]

# 创建3D图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制正方体
for face in cube_faces:
    square = [[cube_vertices[vertex][0], cube_vertices[vertex][1], cube_vertices[vertex][2]] for vertex in face]
    square.append(square[0])  # 闭合图形
    ax.plot(*zip(*square), color='blue')

# 绘制四棱锥，每个使用不同的颜色
colors = ['red', 'blue', 'blue', 'orange']
for vertices, faces, color in zip(
    [pyramid1_vertices, pyramid2_vertices, pyramid3_vertices, pyramid4_vertices],
    [pyramid1_faces, pyramid2_faces, pyramid3_faces, pyramid4_faces],
    colors
):
    for face in faces:
        polygon = [[vertices[vertex][0], vertices[vertex][1], vertices[vertex][2]] for vertex in face]
        polygon.append(polygon[0])  # 闭合图形
        ax.plot(*zip(*polygon), color=color)

# 在每个顶点上添加序号和坐标
# 首先，合并所有顶点并去除重复项
all_vertices = cube_vertices + pyramid1_vertices + pyramid2_vertices + pyramid3_vertices + pyramid4_vertices
unique_vertices = list(set([tuple(v) for v in all_vertices]))  # 去除重复顶点

# 创建一个字典来映射顶点坐标到序号
vertex_to_index = {tuple(v): i for i, v in enumerate(unique_vertices)}

# 标记每个顶点
for vertex in all_vertices:
    index = vertex_to_index[tuple(vertex)]
    ax.text(vertex[0], vertex[1], vertex[2], f'({index}) {vertex}', size=15, zorder=1)

# 设置坐标轴标签
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形

savepath=os.path.join(desktop_path, '飞机模型'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir(savepath)

plt.savefig(os.path.join(savepath,'3d_geometric_shapes.png'), format='png')
plt.savefig(os.path.join(savepath,'3d_geometric_shapes.pdf'), format='pdf')
#plt.show()




def rotate(angle):
    ax.view_init(elev=15, azim=angle)

anim = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 3))
#保存文件
anim.save(os.path.join(savepath,'plane_model_3D.gif'), writer='imagemagick', fps=240)
