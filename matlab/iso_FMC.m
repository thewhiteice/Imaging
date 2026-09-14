%% ===== 仿真参数定义 =====
% 材料定义
SOLID_LAYERS = [
    % struct("name","Cu", "cp",4710, "csh",2260, "th",15e-3, "rho",8900);
    % struct("name","Fe", "cp",6100, "csh",3300, "th",15e-3, "rho",7800);
    struct("name","Al", "cp",6260, "csh",3080, "th",15e-3, "rho",2700);
];

air_th = 0.1 * sum([SOLID_LAYERS.th]);
AIR_LAYER = [
    struct("name", "Air", "cp", 343, "csh", 1, "th", air_th, "rho", 200)
];

LAYERS = [AIR_LAYER; SOLID_LAYERS; AIR_LAYER];

% 仿真空间定义
Lx = 15.e-3;    % m
Ly = sum([LAYERS.th]);  % m

% 源频率定义
f0 = 5.e6;      % Hz
n_cycles = 5;
amp = 1e6;

% 派生参数计算
c_values = [[SOLID_LAYERS.cp], [SOLID_LAYERS.csh]];
c_max = max(c_values); % 最大固体纵/横波声速
c_min = min(c_values); % 最小固体纵/横波声速
lambda_min = c_min / f0; % 最小波长

rho_values = [SOLID_LAYERS.rho];
rho_min = min(rho_values); %最小固体密度

% 换能器位置定义 
% 纵坐标会被实际修改
n_sensors = 8;
dx_sensors = lambda_min / 4;

sensor_x = linspace(0, dx_sensors*n_sensors, n_sensors);
sensor_x = sensor_x + Lx/2;
sensor_y = air_th * ones(1, n_sensors);
sensor_positions = [sensor_x(:), sensor_y(:)];

% 缺陷位置定义
n_points = 0;
r_points_num = 3;
r_points_range = 0: r_points_num-1;
row_range = 0:r_points_num-1;
col_range = 0:r_points_num-1;

Lx_p = Lx;
Ly_p = sum([SOLID_LAYERS.th]);

point_x = linspace(0.25*Lx_p, 0.75*Lx_p, n_points);
point_y = linspace(0.25*Ly_p, 0.75*Ly_p, n_points);

% CFL条件数
cfl = 0.3;

% 材料衰减定义
alpha_coeff_compression = 0.1;
alpha_coeff_shear = 0.5;

% PML定义
pml_alpha = [2,2];
PML_size = 10;

% 网格尺寸定义
ppw = 6;     % 每波长单元数

% 网格间距设定
dx = c_min / (f0 * ppw);
Nx = 2^nextpow2((round(Lx / dx)));
Ny = 2^nextpow2((round(Ly / dx)));

dx = Lx / Nx;
dy = Ly / Ny;

%% ===== 换能器坐标修正 =====
sensor_x = round(sensor_positions(:,1)/dx)+1;
sensor_y = round(sensor_positions(:,2)/dy)+5;

sensor_positions = [(sensor_x - 1) * dx, (sensor_y - 1) * dy];

sensor_pos_save = sensor_positions;
sensor_pos_save(:,2) = sensor_pos_save(:,2) - air_th;   % 平移到固体顶面为 y=0

%% ===== 仿真网格设置 =====
kgrid=kWaveGrid(Nx,dx,Ny,dy);

t_end = 2 * sqrt(Lx_p^2 + Ly_p^2) / c_min;
% t_end = sum([SOLID_LAYERS.th] ./ [SOLID_LAYERS.cp]);
kgrid.makeTime(c_max, cfl, t_end);

t_axis = kgrid.t_array;

%% ===== 声速矩阵定义 =====
cp_matrix = zeros(Nx, Ny);
csh_matrix = zeros(Nx, Ny);
rho_matrix = zeros(Nx, Ny);

n_rows = zeros(length(LAYERS), 1);
for i = 1:length(LAYERS)
    n_rows(i) = round(LAYERS(i).th / dy);
end

diff_y = Ny - sum(n_rows);
n_rows(end) = n_rows(end) + diff_y;

ny_start = 1;
layer_indices = cell(length(LAYERS), 1);

for i = 1:length(LAYERS)
    ny_end = ny_start + n_rows(i) - 1;

    if i == length(LAYERS)
        ny_end = Ny;
    end
    layer_indices{i} = [ny_start, ny_end];

    cp_matrix(:, ny_start:ny_end) = LAYERS(i).cp;
    csh_matrix(:, ny_start:ny_end) = LAYERS(i).csh;
    rho_matrix(:, ny_start:ny_end) = LAYERS(i).rho;
    layer_indices{i} = [ny_start, ny_end];

    ny_start = ny_end + 1;
end

% 定义散射体位置
p_i = round(point_x/dx)+1;  % 数组索引
p_j = round(point_y/dx)+1;  % 数组索引

for k = 1:length(p_i)
    % 当前散射体的矩形块范围
    rows = p_i(k) + row_range;
    cols = p_j(k) + col_range;

    % 一次性赋值整个矩形块（不用再内层循环）
    cp_matrix(rows, cols)  = c_min * 0.1;
    csh_matrix(rows, cols) = min([SOLID_LAYERS.csh]) * 0.1;
    rho_matrix(rows, cols) = rho_min * 0.1;
end

medium.sound_speed_compression = cp_matrix;
medium.sound_speed_shear = csh_matrix;
medium.density = rho_matrix;
% medium.alpha_coeff_compression = alpha_coeff_compression;
% medium.alpha_coeff_shear = alpha_coeff_shear;

%% ===== 传感器设置 =====
sensor.mask = zeros(Nx, Ny);
for i = 1:size(sensor_positions, 1)
    ix = round(sensor_positions(i,1)/dx) + 1;
    iy = round(sensor_positions(i,2)/dy) + 1;
    sensor.mask(ix, iy) = 1;
end

sensor.record = {'u'};

%% ===== 进行全矩阵采集循环 =====
% 传感器数据 FMC
sensor_data_fmc = zeros(n_sensors, n_sensors, length(t_axis));
source.s_mask = zeros(Nx, Ny);

for tx_idx = 1: n_sensors
    fprintf("开启第%d轮仿真\n",tx_idx);
    % 激发源设置 
    source_x = round(sensor_positions(tx_idx,1)/dx)+1;
    source_y = round(sensor_positions(tx_idx,2)/dy)+1;
    source.s_mask(source_x, source_y) = 1;
    
    signal = 1.0 * toneBurst(1./kgrid.dt, f0, n_cycles); % 默认高斯包络
    source.sxx = 0.0;
    source.syy = signal;

    % 执行仿真 
    sensor_data = pstdElastic2D(kgrid, medium, source, sensor, ...
        'PMLAlpha', pml_alpha, ...
        'PMLSize', [PML_size, PML_size], ...
        'PlotSim', false, 'PlotPML', false);

   % 记录数据
   sensor_data_fmc(tx_idx,: ,:) = sensor_data.uy;
   fprintf("第%d轮仿真已完成\n",tx_idx);
   fprintf("\n");
end

%% ===== 可视化 =====
figure;
hold on;
imagesc(cp_matrix');
for i = 1:n_sensors
    plot(sensor_x(i), sensor_y(i), 'r*', 'MarkerSize', 3, 'LineWidth', 2);
end
colorbar;
hold off;
title('纵波速度分布');
xlabel('x 方向网格点');   % 注意：imagesc 默认 x 轴是列索引，对应 y
ylabel('y 方向网格点');   % y 轴是行索引，对应 x
axis equal tight;

figure;
hold on;
for i = 1:size(sensor_data.uy, 1)
    plot(t_axis, sensor_data.uy(i,:), 'DisplayName', sprintf('传感器 %d', i));
end
legend show;
xlabel('时间 (s)');
ylabel('位移 uy(m) ');
grid on;
hold off;

% ===== 频谱分析 =====
% 绘制激励信号频谱
Y_sig = fft(sensor_data.uy, [], 2);
signal_padded = [signal(:); zeros(length(t_axis)-length(signal), 1)];
Y = fft(signal_padded);
L = size(t_axis, 2); 
N = floor(L/2) + 1;   % 正频率部分
Y_sig = Y_sig(:, 1:N);
Y = Y(1:N);

% --- 计算单边幅度谱 ---
P1 = abs(Y_sig/L);      % 双边谱的幅度（归一化）
P1 = P1(:, 1:N);    % 取正频率部分
P1(:, 2:end-1) = 2*P1(:, 2:end-1); % 除了直流和奈奎斯特频率外，幅值乘以2

P2 = abs(Y/L);  % 双边谱的幅度（归一化）
P2 = P2(1:N);   % 取正频率部分
P2(2:end-1) = 2*P2(2:end-1); % 除了直流和奈奎斯特频率外，幅值乘以2

% --- 生成频率轴 ---
dt = mean(diff(t_axis));   % 时间间隔平均值
Fs = 1 / dt;               % 采样频率
f = Fs*(0:N-1)/L;   % 单边谱对应的频率范围 (0 到 Nyquist)

P1_norm = P1 ./ max(P1(:));   
P2_norm = P2 ./ max(P2(:)); 

figure;
hold on;
for i = 1:size(P1, 1)
    plot(f, P1_norm(i,:), 'DisplayName', sprintf('传感器 %d', i), 'LineWidth', 1.5);
end
plot(f, P2_norm, 'DisplayName', '激励信号', 'LineWidth', 1.5);
hold off
title('单边幅度谱');
xlabel('频率 (Hz)');
ylabel('幅度');
legend show; 
grid on;
xlim([0 10.0e6]);       % 限制显示范围到奈奎斯特频率

%% ===== 保存文件 =====
% 1. 获取 root 路径（自动识别）
scriptDir = pwd;                        % 返回 /path/to/root/scripts
rootDir = fileparts(scriptDir);         % 返回 /path/to/root
dataDir = fullfile(rootDir, 'data');    % /path/to/root/data

% 如果 data 目录不存在则创建
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% 2. 定义总参数（用作子文件夹名）
main_param = sprintf('iso_%dL_%.1fMhz_%dsensor_%dpoint_FMC', ...
             length(SOLID_LAYERS), f0/1.e6, n_sensors, n_points);
% 若担心重名，可追加时间戳
main_param = [main_param '_' char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))];
saveFolder = fullfile(dataDir, main_param);
if ~exist(saveFolder, 'dir')
    mkdir(saveFolder);
end

% 3. 定义所有参数和数据
params.Lx = Lx;                         % 仿真区域
params.Ly = sum([SOLID_LAYERS.th]);     % 仿真区域
params.f0 = f0;                         % 源频率
params.lambda = lambda_min;             % 最小波长
params.cp = [SOLID_LAYERS.cp];          % 纵波序列
params.t_axis = t_axis;                 % 仿真时间
params.fmc_data = sensor_data_fmc;      % 换能器信号
params.sensor_pos = sensor_pos_save;    % 换能器位置

% 4. 遍历结构体，每个字段存为一个 .bin
fields = fieldnames(params);
for i = 1:length(fields)
    field_name = fields{i};
    data_value = params.(field_name);

    fid = fopen(fullfile(saveFolder, [field_name '.bin']), 'wb');

    if ischar(data_value) || isstring(data_value)
        % 字符串按字节流写入，末尾加0作为结束标志
        str_bytes = uint8(char(data_value));
        fwrite(fid, str_bytes, 'uint8');
        fwrite(fid, 0, 'uint8');   % 字符串结束标志
    else
        % 数值数据：写入维度信息(维度数和各维度大小), 本体
        dims = size(data_value);
        fwrite(fid, ndims(data_value), 'int32');  % 维度数
        fwrite(fid, dims, 'int32');               % 各维度大小（列优先顺序与写入一致）
        fwrite(fid, data_value, 'double');
    end
    fclose(fid);
end

disp(['数据已保存到: ' saveFolder]);
disp(['当前工作目录: ' pwd]);
disp(['脚本所在目录: ' scriptDir]);
