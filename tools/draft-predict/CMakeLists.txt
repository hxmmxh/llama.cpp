set(TARGET llama-draft-predict)
add_executable(${TARGET} draft-predict.cpp)
target_link_libraries(${TARGET} PRIVATE llama-speculative-adaptive vendor::nlohmann)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
