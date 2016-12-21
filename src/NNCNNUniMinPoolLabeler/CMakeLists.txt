
include_directories(
  model
)

add_executable(NNCNNUniMinPoolLabeler NNCNNUniMinPoolLabeler.cpp)
target_link_libraries(NNCNNUniMinPoolLabeler ${LIBS})
