
include_directories(
  model
)

add_executable(NNCNNTriPoolLabeler NNCNNTriPoolLabeler.cpp)
target_link_libraries(NNCNNTriPoolLabeler ${LIBS})
