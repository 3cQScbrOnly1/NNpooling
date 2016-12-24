
include_directories(
  model
)

add_executable(NNCNNUniMinReluPoolLabeler NNCNNUniMinReluPoolLabeler.cpp)
target_link_libraries(NNCNNUniMinReluPoolLabeler ${LIBS})
