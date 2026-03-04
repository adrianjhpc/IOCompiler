# File write benchmark

# Standard -O3 Baseline
add_executable(pixel_write_standard pixel_write_benchmark.cpp)
target_compile_options(pixel_write_standard PRIVATE -O3)

# Custom Pass -O3 (Optimised)
add_executable(pixel_write_fast pixel_write_benchmark.cpp)

# Use the generator expression $<TARGET_FILE:IOOpt> to get the absolute path
# to your library automatically. This also creates a build dependency
target_compile_options(pixel_write_fast PRIVATE 
    -O3 
    "-fpass-plugin=$<TARGET_FILE:IOOpt>"
)

# Ensure bench_fast only builds after the plugin is ready
add_dependencies(pixel_write_fast IOOpt)

# HTTP Server benchmark

#Standard -O3 Baseline
add_executable(http_standard http_benchmark.cpp)
target_compile_options(http_standard PRIVATE -O3 -fno-inline)

# Custom Pass -O3 (Optimised)
add_executable(http_fast http_benchmark.cpp)
# Load your custom IOOpt pass
target_compile_options(http_fast PRIVATE 
    -O3 
    -fno-inline 
    -fpass-plugin=$<TARGET_FILE:IOOpt>
)
# Ensure the pass is built before compiling this benchmark
add_dependencies(http_fast IOOpt)

# Database WAL benchmark

# Baseline
add_executable(wal_standard database_wal_benchmark.cpp)
target_compile_options(wal_standard PRIVATE -O3 -fno-inline)

# Optimised
add_executable(wal_fast database_wal_benchmark.cpp)
target_compile_options(wal_fast PRIVATE -O3 -fno-inline -fpass-plugin=$<TARGET_FILE:IOOpt>)
add_dependencies(wal_fast IOOpt)

# Multimedia streaming benchmark

# Baseline
add_executable(media_standard media_benchmark.cpp)
target_compile_options(media_standard PRIVATE -O3 -fno-inline)

# Optimised
add_executable(media_fast media_benchmark.cpp)
target_compile_options(media_fast PRIVATE -O3 -fno-inline -fpass-plugin=$<TARGET_FILE:IOOpt>)
add_dependencies(media_fast IOOpt)

# LTO cross-file benchmark

# Baseline
add_executable(lto_standard 
    lto_bench_main.cpp 
    lto_bench_log.cpp
)
target_compile_options(lto_standard PRIVATE -O3)

# Optimised: LTO enabled and Custom Pass Plugin
add_executable(lto_fast 
    lto_bench_main.cpp 
    lto_bench_log.cpp
)

# Load the plugin during compile and link
target_compile_options(lto_fast PRIVATE -O3 -flto=full -fpass-plugin=$<TARGET_FILE:IOOpt>)
target_link_options(lto_fast PRIVATE -O3 -flto=full -fuse-ld=lld "-Wl,--load-pass-plugin=$<TARGET_FILE:IOOpt>")

add_dependencies(lto_fast IOOpt)


