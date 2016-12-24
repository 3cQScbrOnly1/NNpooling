
include_directories(
  model
)

add_executable(NNCNNTriSigPoolLabeler NNCNNTriSigPoolLabeler.cpp)
target_link_libraries(NNCNNTriSigPoolLabeler ${LIBS})
