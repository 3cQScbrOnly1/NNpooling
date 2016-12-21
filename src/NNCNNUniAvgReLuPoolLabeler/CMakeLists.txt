
include_directories(
  model
)

add_executable(NNCNNUniAvgReLuPoolLabeler NNCNNUniAvgReLuPoolLabeler.cpp)
target_link_libraries(NNCNNUniAvgReLuPoolLabeler ${LIBS})
