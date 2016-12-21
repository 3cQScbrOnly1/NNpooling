
include_directories(
  model
)

add_executable(NNpooling NNpooling.cpp)
target_link_libraries(NNpooling ${LIBS})
