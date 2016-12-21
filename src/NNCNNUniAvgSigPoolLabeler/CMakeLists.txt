
include_directories(
  model
)

add_executable(NNCNNUniAvgSigPoolLabeler NNCNNUniAvgSigPoolLabeler.cpp)
target_link_libraries(NNCNNUniAvgSigPoolLabeler ${LIBS})
