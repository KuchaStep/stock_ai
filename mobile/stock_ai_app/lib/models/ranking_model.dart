class RankingModel {
  final String code;
  final String name;
  final DateTime date;
  final double close;
  final double upProbability;

  RankingModel({
    required this.code,
    required this.name,
    required this.date,
    required this.close,
    required this.upProbability,
  });

  factory RankingModel.fromJson(Map<String, dynamic> json) {
    return RankingModel(
      code: json["code"],
      name: json["name"],
      date: DateTime.parse(json["date"]),
      close: (json["close"] as num).toDouble(),
      upProbability: (json["up_probability"] as num).toDouble(),
    );
  }
}