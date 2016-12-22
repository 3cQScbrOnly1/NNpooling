
include_directories(
  model
)

add_executable(NNCNNUniMaxReluPoolLabeler NNCNNUniMaxReluPoolLabeler.cpp)
target_link_libraries(NNCNNUniMaxReluPoolLabeler ${LIBS})
