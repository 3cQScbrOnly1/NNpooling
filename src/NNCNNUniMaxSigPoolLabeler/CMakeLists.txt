
include_directories(
  model
)

add_executable(NNCNNUniMaxSigPoolLabeler NNCNNUniMaxSigPoolLabeler.cpp)
target_link_libraries(NNCNNUniMaxSigPoolLabeler ${LIBS})
