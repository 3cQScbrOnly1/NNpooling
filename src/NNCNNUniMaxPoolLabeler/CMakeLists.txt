
include_directories(
  model
)

add_executable(NNCNNUniMaxPoolLabeler NNCNNUniMaxPoolLabeler.cpp)
target_link_libraries(NNCNNUniMaxPoolLabeler ${LIBS})
