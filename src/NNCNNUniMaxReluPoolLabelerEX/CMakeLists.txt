
include_directories(
  model
)

add_executable(NNCNNUniMaxReluPoolLabelerEX NNCNNUniMaxReluPoolLabeler.cpp)
target_link_libraries(NNCNNUniMaxReluPoolLabelerEX ${LIBS})
