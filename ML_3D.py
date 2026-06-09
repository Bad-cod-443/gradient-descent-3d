import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. МАТЕМАТИКА: Функция и её градиент
# ==========================================
def f(x, y):
    # Сама 3D поверхность (холмы и впадины)
    return x**2 + y**2 + 5 * np.sin(x) + 5 * np.cos(y)

def grad_f(x, y):
    # Производные (указывают направление уклона)
    dx = 2 * x + 5 * np.cos(x)
    dy = 2 * y - 5 * np.sin(y)
    return dx, dy

# 2. ПОДГОТОВКА ЛАНДШАФТА (СЕТКА)

x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = f(X, Y)

# Создаем пустой холст и сразу кладем на него поверхность
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])


# 3. БАЗОВЫЕ НАСТРОЙКИ (Гиперпараметры)

learning_rate = 0.05
epochs = 50
start_x, start_y = 4.0, 4.0 # Точка старта для всех шариков


# 4. ГОНЩИК №1: ОБЫЧНЫЙ ГРАДИЕНТНЫЙ СПУСК (Красный)

cur_x_gd, cur_y_gd = start_x, start_y
path_x_gd, path_y_gd, path_z_gd = [cur_x_gd], [cur_y_gd], [f(cur_x_gd, cur_y_gd)]

for i in range(epochs):
    dx, dy = grad_f(cur_x_gd, cur_y_gd)
    
    cur_x_gd = cur_x_gd - learning_rate * dx
    cur_y_gd = cur_y_gd - learning_rate * dy
    
    path_x_gd.append(cur_x_gd)
    path_y_gd.append(cur_y_gd)
    path_z_gd.append(f(cur_x_gd, cur_y_gd))

# Рисуем путь красного шарика
fig.add_trace(go.Scatter3d(
    x=path_x_gd, y=path_y_gd, z=path_z_gd,
    mode='lines+markers', marker=dict(size=4, color='red'), line=dict(color='red', width=4),
    name='Vanilla GD'
))

# ==========================================
# 5. ГОНЩИК №2: MOMENTUM (Синий)
# ==========================================
cur_x_mom, cur_y_mom = start_x, start_y
path_x_mom, path_y_mom, path_z_mom = [cur_x_mom], [cur_y_mom], [f(cur_x_mom, cur_y_mom)]

v_x, v_y = 0.0, 0.0  
gamma = 0.9 # Коэффициент трения/инерции

for i in range(epochs):
    dx, dy = grad_f(cur_x_mom, cur_y_mom)
    
    v_x = gamma * v_x + learning_rate * dx
    v_y = gamma * v_y + learning_rate * dy
    
    cur_x_mom = cur_x_mom - v_x
    cur_y_mom = cur_y_mom - v_y
    
    path_x_mom.append(cur_x_mom)
    path_y_mom.append(cur_y_mom)
    path_z_mom.append(f(cur_x_mom, cur_y_mom))

# Рисуем путь синего шарика
fig.add_trace(go.Scatter3d(
    x=path_x_mom, y=path_y_mom, z=path_z_mom,
    mode='lines+markers', marker=dict(size=4, color='blue'), line=dict(color='blue', width=4),
    name='Momentum'
))


# 6. ГОНЩИК №3: ADAM (Зеленый)

cur_x_adam, cur_y_adam = start_x, start_y
path_x_adam, path_y_adam, path_z_adam = [cur_x_adam], [cur_y_adam], [f(cur_x_adam, cur_y_adam)]

m_x, m_y = 0.0, 0.0
v_x_adam, v_y_adam = 0.0, 0.0
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8

for i in range(1, epochs + 1):
    dx, dy = grad_f(cur_x_adam, cur_y_adam)
    
    # Накопление инерции
    m_x = beta1 * m_x + (1 - beta1) * dx
    m_y = beta1 * m_y + (1 - beta1) * dy
    
    # Накопление квадратов градиента
    v_x_adam = beta2 * v_x_adam + (1 - beta2) * (dx**2)
    v_y_adam = beta2 * v_y_adam + (1 - beta2) * (dy**2)
    
    m_x_corr = m_x / (1 - beta1**i)
    m_y_corr = m_y / (1 - beta1**i)
    v_x_corr = v_x_adam / (1 - beta2**i)
    v_y_corr = v_y_adam / (1 - beta2**i)
    
    # Умный шаг
    cur_x_adam = cur_x_adam - learning_rate * m_x_corr / (np.sqrt(v_x_corr) + epsilon)
    cur_y_adam = cur_y_adam - learning_rate * m_y_corr / (np.sqrt(v_y_corr) + epsilon)
    
    path_x_adam.append(cur_x_adam)
    path_y_adam.append(cur_y_adam)
    path_z_adam.append(f(cur_x_adam, cur_y_adam))

# Рисуем путь зеленого шарика
fig.add_trace(go.Scatter3d(
    x=path_x_adam, y=path_y_adam, z=path_z_adam,
    mode='lines+markers', marker=dict(size=4, color='green'), line=dict(color='green', width=4),
    name='Adam'
))

# 7. ФИНАЛЬНЫЕ НАСТРОЙКИ ГРАФИКА И ВЫВОД

fig.update_layout(
    title='Гонка оптимизаторов: Vanilla GD vs Momentum vs Adam',
    width=1000,
    height=800,
    scene=dict(
        xaxis_title='Вес X',
        yaxis_title='Вес Y',
        zaxis_title='Ошибка Z (Loss)'
    )
)

fig.show()
