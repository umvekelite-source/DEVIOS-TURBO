set(TARGET devios-turbo)

add_executable(${TARGET}
    main.cpp
    devios_corpus.cpp
    devios_drafter.cpp
    devios_engine.cpp
    devios_server.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib
    ${CMAKE_SOURCE_DIR}/vendor/nlohmann
)

target_link_libraries(${TARGET} PRIVATE
    llama-common
    llama
    cpp-httplib
    Threads::Threads
)

if (MSVC)
    target_compile_options(${TARGET} PRIVATE
        /O2 /Oi /Ot /arch:AVX2 /GL /Gy /Qpar
    )
    target_link_options(${TARGET} PRIVATE
        /LTCG /OPT:REF /OPT:ICF
    )
endif()

add_custom_command(TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${TARGET}>
        $<TARGET_FILE_DIR:${TARGET}>/devios-turbo-extreme.exe
    COMMENT "Duplicating devios-turbo to devios-turbo-extreme.exe"
)

