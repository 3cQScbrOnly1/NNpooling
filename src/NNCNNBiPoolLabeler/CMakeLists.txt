
include_directories(
  model
)

add_executable(NNCNNBiPoolLabeler NNCNNBiPoolLabeler.cpp)
target_link_libraries(NNCNNBiPoolLabeler ${LIBS})
