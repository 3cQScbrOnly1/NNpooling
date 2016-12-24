
include_directories(
  model
)

add_executable(NNCNNTriReluPoolLabeler NNCNNTriReluPoolLabeler.cpp)
target_link_libraries(NNCNNTriReluPoolLabeler ${LIBS})
