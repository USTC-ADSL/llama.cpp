set(TARGET llama-system-benefit-planner)

add_executable(${TARGET} system-benefit-planner.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
