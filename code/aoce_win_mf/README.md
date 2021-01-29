# aoce_win_mf

## 注意点

MF 异步模式每次读取的数据可能并不在同一线程上,这样可能导致问题,如vulkan gpu可能分散在不同线程reset.
