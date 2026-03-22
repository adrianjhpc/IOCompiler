# Find the Clang executable inside your custom LLVM installation
find_program(CLANG_EXE clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)

if(NOT CLANG_EXE)
    message(WARNING "Could not find 'clang' in the LLVM installation. Tests will fail.")
endif()

# Define the test
add_test(NAME AutoZeroCopyTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_zerocopy_test.sh
            ${CLANG_EXE}                     
            $<TARGET_FILE:io-opt>            
            ${CMAKE_CURRENT_SOURCE_DIR}/zerocopy_test.c
)

set_tests_properties(AutoZeroCopyTest PROPERTIES TIMEOUT 10)

add_test(NAME VectoredTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_vectored_test.sh
            ${CLANG_EXE}                    
            $<TARGET_FILE:io-opt>            
            ${CMAKE_CURRENT_SOURCE_DIR}/vectored_test.c
)

set_tests_properties(AutoZeroCopyTest PROPERTIES TIMEOUT 10)


add_test(NAME AsyncTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_async_test.sh
            ${CLANG_EXE}
            $<TARGET_FILE:io-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/async_test.c
)

set_tests_properties(AsyncTest PROPERTIES TIMEOUT 10)

add_test(NAME MMapTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_mmap_test.sh
            ${CLANG_EXE}
            $<TARGET_FILE:io-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/mmap_test.c
)

set_tests_properties(MMapTest PROPERTIES TIMEOUT 10)

add_test(NAME PrefetchTest
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_prefetch_test.sh
            ${CLANG_EXE}
            $<TARGET_FILE:io-opt>
            ${CMAKE_CURRENT_SOURCE_DIR}/prefetch_test.c
)

set_tests_properties(PrefetchTest PROPERTIES TIMEOUT 10)

