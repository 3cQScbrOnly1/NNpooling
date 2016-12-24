
include_directories(
  model
)

add_executable(NNCNNUniMinTanhPoolLabeler NNCNNUniMinTanhPoolLabeler.cpp)
target_link_libraries(NNCNNUniMinTanhPoolLabeler ${LIBS})
