#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/wifi-module.h"
#include "ns3/energy-module.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("ManetOpportunisticMeanFieldRouting");

class OpportunisticHeader : public Header
{
public:
  static constexpr uint32_t MAX_CANDIDATES = 8;

  OpportunisticHeader()
      : m_src(0),
        m_dst(0),
        m_packetId(0),
        m_previousHop(0),
        m_hopCount(0),
        m_ttl(0),
        m_candidateCount(0)
  {
    m_candidates.fill(0);
  }

  void SetBase(uint32_t src, uint32_t dst, uint32_t packetId, uint32_t previousHop, uint32_t hopCount, int32_t ttl)
  {
    m_src = src;
    m_dst = dst;
    m_packetId = packetId;
    m_previousHop = previousHop;
    m_hopCount = hopCount;
    m_ttl = ttl;
  }

  uint32_t GetSrc() const
  {
    return m_src;
  }

  uint32_t GetDst() const
  {
    return m_dst;
  }

  uint32_t GetPacketId() const
  {
    return m_packetId;
  }

  uint32_t GetPreviousHop() const
  {
    return m_previousHop;
  }

  uint32_t GetHopCount() const
  {
    return m_hopCount;
  }

  int32_t GetTtl() const
  {
    return m_ttl;
  }

  void SetPreviousHop(uint32_t id)
  {
    m_previousHop = id;
  }

  void SetHopCount(uint32_t count)
  {
    m_hopCount = count;
  }

  void SetTtl(int32_t ttl)
  {
    m_ttl = ttl;
  }

  void SetCandidates(const std::vector<uint32_t>& candidates)
  {
    m_candidateCount = std::min<uint32_t>(MAX_CANDIDATES, candidates.size());
    for (uint32_t i = 0; i < MAX_CANDIDATES; ++i)
      {
        m_candidates[i] = (i < m_candidateCount) ? candidates[i] : 0;
      }
  }

  uint32_t GetCandidateCount() const
  {
    return m_candidateCount;
  }

  std::vector<uint32_t> GetCandidates() const
  {
    std::vector<uint32_t> out;
    out.reserve(m_candidateCount);
    for (uint32_t i = 0; i < m_candidateCount; ++i)
      {
        out.push_back(m_candidates[i]);
      }
    return out;
  }

  bool IsCandidate(uint32_t nodeId) const
  {
    for (uint32_t i = 0; i < m_candidateCount; ++i)
      {
        if (m_candidates[i] == nodeId)
          {
            return true;
          }
      }
    return false;
  }

  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::OpportunisticHeader")
                            .SetParent<Header>()
                            .AddConstructor<OpportunisticHeader>();
    return tid;
  }

  TypeId GetInstanceTypeId() const override
  {
    return GetTypeId();
  }

  void Serialize(Buffer::Iterator i) const override
  {
    i.WriteHtonU32(m_src);
    i.WriteHtonU32(m_dst);
    i.WriteHtonU32(m_packetId);
    i.WriteHtonU32(m_previousHop);
    i.WriteHtonU32(m_hopCount);
    i.WriteHtonU32(static_cast<uint32_t>(m_ttl));
    i.WriteHtonU32(m_candidateCount);
    for (uint32_t iCandidate = 0; iCandidate < MAX_CANDIDATES; ++iCandidate)
      {
        i.WriteHtonU32(m_candidates[iCandidate]);
      }
  }

  uint32_t Deserialize(Buffer::Iterator i) override
  {
    m_src = i.ReadNtohU32();
    m_dst = i.ReadNtohU32();
    m_packetId = i.ReadNtohU32();
    m_previousHop = i.ReadNtohU32();
    m_hopCount = i.ReadNtohU32();
    m_ttl = static_cast<int32_t>(i.ReadNtohU32());
    m_candidateCount = std::min<uint32_t>(MAX_CANDIDATES, i.ReadNtohU32());
    for (uint32_t iCandidate = 0; iCandidate < MAX_CANDIDATES; ++iCandidate)
      {
        m_candidates[iCandidate] = i.ReadNtohU32();
      }
    return GetSerializedSize();
  }

  uint32_t GetSerializedSize() const override
  {
    return (7 + MAX_CANDIDATES) * sizeof(uint32_t);
  }

  void Print(std::ostream& os) const override
  {
    os << "src=" << m_src << " dst=" << m_dst << " pkt=" << m_packetId << " prev=" << m_previousHop
       << " hop=" << m_hopCount << " ttl=" << m_ttl << " cands=" << m_candidateCount;
  }

private:
  uint32_t m_src;
  uint32_t m_dst;
  uint32_t m_packetId;
  uint32_t m_previousHop;
  uint32_t m_hopCount;
  int32_t m_ttl;
  uint32_t m_candidateCount;
  std::array<uint32_t, MAX_CANDIDATES> m_candidates;
};

class MeanFieldLearner
{
public:
  MeanFieldLearner()
      : m_alpha(0.2),
        m_gamma(0.9),
        m_beta(0.35),
        m_epsilon(0.2),
        m_epsilonBase(0.08),
        m_mobilityWeight(0.35),
        m_varianceWeight(0.30),
        m_dropWeight(0.35),
        m_mobilitySignal(0.0),
        m_rewardVarianceSignal(0.0),
        m_dropRateSignal(0.0),
        m_minEpsilon(0.05),
        m_maxEpsilon(0.50),
        m_epsilonDecayFactor(120.0),
        m_policyMomentum(0.95),
        m_rng(CreateObject<UniformRandomVariable>())
  {
  }

  void ConfigureHybridEpsilon(double base,
                              double mobilityWeight,
                              double varianceWeight,
                              double dropWeight,
                              double minEpsilon,
                              double maxEpsilon,
                              double decayFactor)
  {
    m_epsilonBase = std::max(0.0, base);
    m_mobilityWeight = std::max(0.0, mobilityWeight);
    m_varianceWeight = std::max(0.0, varianceWeight);
    m_dropWeight = std::max(0.0, dropWeight);
    m_minEpsilon = std::max(0.0, minEpsilon);
    m_maxEpsilon = std::max(m_minEpsilon, maxEpsilon);
    m_epsilonDecayFactor = std::max(1.0, decayFactor);
  }

  void SetHybridSignals(double mobility, double rewardVariance, double dropRate)
  {
    m_mobilitySignal = std::clamp(mobility, 0.0, 1.0);
    m_rewardVarianceSignal = std::clamp(rewardVariance, 0.0, 1.0);
    m_dropRateSignal = std::clamp(dropRate, 0.0, 1.0);
  }

  double ComputeEpsilon(double nowSeconds) const
  {
    const double baseHybrid = m_epsilonBase + (m_mobilityWeight * m_mobilitySignal) +
                              (m_varianceWeight * m_rewardVarianceSignal) + (m_dropWeight * m_dropRateSignal);
    const double decayed = baseHybrid * std::exp(-nowSeconds / m_epsilonDecayFactor);
    return std::clamp(decayed, m_minEpsilon, m_maxEpsilon);
  }

  double GetQValue(uint32_t node, uint32_t action) const
  {
    auto itNode = m_q.find(node);
    if (itNode == m_q.end())
      {
        return 0.0;
      }
    auto itAction = itNode->second.find(action);
    return (itAction == itNode->second.end()) ? 0.0 : itAction->second;
  }

  uint32_t SelectAction(uint32_t node,
                        const std::vector<uint32_t>& candidates,
                        uint32_t destination,
                        const NodeContainer& nodes,
                        double commRange)
  {
    if (candidates.empty())
      {
        return node;
      }

    const double epsHybrid = ComputeEpsilon(Simulator::Now().GetSeconds());
    m_epsilon = epsHybrid;

    const double randomSample = m_rng->GetValue(0.0, 1.0);
    if (randomSample < epsHybrid)
      {
        const uint32_t idx = static_cast<uint32_t>(m_rng->GetInteger(0, candidates.size() - 1));
        const uint32_t chosen = candidates[idx];
        UpdatePolicy(node, candidates, chosen);
        return chosen;
      }

    double sumQ = 0.0;
    for (auto c : candidates)
      {
        sumQ += m_q[node][c];
      }
    const double meanQ = sumQ / static_cast<double>(candidates.size());

    const auto nodeMob = nodes.Get(node)->GetObject<MobilityModel>();
    const auto dstMob = nodes.Get(destination)->GetObject<MobilityModel>();
    const double myDist = nodeMob->GetDistanceFrom(dstMob);

    double bestScore = -std::numeric_limits<double>::max();
    uint32_t best = candidates.front();

    for (auto c : candidates)
      {
        const auto cMob = nodes.Get(c)->GetObject<MobilityModel>();
        const double cDist = cMob->GetDistanceFrom(dstMob);
        const double progress = (myDist - cDist) / std::max(1.0, commRange);

        const double score = m_q[node][c] + (m_beta * meanQ) + (0.55 * progress) + (0.10 * m_policy[node][c]);

        if (score > bestScore)
          {
            bestScore = score;
            best = c;
          }
      }

    UpdatePolicy(node, candidates, best);
    return best;
  }

  void UpdateQ(uint32_t node,
               uint32_t action,
               double reward,
               uint32_t nextNode,
               const std::vector<uint32_t>& nextCandidates)
  {
    double maxNext = 0.0;
    double meanNext = 0.0;

    if (!nextCandidates.empty())
      {
        maxNext = -std::numeric_limits<double>::max();
        for (auto nextAction : nextCandidates)
          {
            const double value = m_q[nextNode][nextAction];
            maxNext = std::max(maxNext, value);
            meanNext += value;
          }
        meanNext /= static_cast<double>(nextCandidates.size());
      }

    const double oldQ = m_q[node][action];
    const double meanFieldBootstrap = ((1.0 - m_beta) * maxNext) + (m_beta * meanNext);
    const double target = reward + (m_gamma * meanFieldBootstrap);
    m_q[node][action] = oldQ + m_alpha * (target - oldQ);

    // epsilon is controlled by hybrid signals each action step
  }

private:
  void UpdatePolicy(uint32_t node, const std::vector<uint32_t>& actions, uint32_t chosen)
  {
    for (auto action : actions)
      {
        m_policy[node][action] *= m_policyMomentum;
      }

    m_policy[node][chosen] = (m_policy[node][chosen] * m_policyMomentum) + (1.0 - m_policyMomentum);
  }

  double m_alpha;
  double m_gamma;
  double m_beta;
  double m_epsilon;
  double m_epsilonBase;
  double m_mobilityWeight;
  double m_varianceWeight;
  double m_dropWeight;
  double m_mobilitySignal;
  double m_rewardVarianceSignal;
  double m_dropRateSignal;
  double m_minEpsilon;
  double m_maxEpsilon;
  double m_epsilonDecayFactor;
  double m_policyMomentum;

  std::map<uint32_t, std::map<uint32_t, double>> m_q;
  std::map<uint32_t, std::map<uint32_t, double>> m_policy;
  Ptr<UniformRandomVariable> m_rng;
};

static NodeContainer g_nodes;
static std::vector<Ptr<Socket>> g_sockets;
static std::vector<std::set<uint64_t>> g_seenPackets;
static std::vector<std::map<uint64_t, EventId>> g_pendingForwards;
static std::vector<Ptr<energy::EnergySource>> g_energySources;
static MeanFieldLearner g_learner;

struct NodeLocalStats
{
  uint32_t selected = 0;
  uint32_t dropped = 0;
  std::deque<double> rewards;
  double rewardVariance = 0.0;
};

static std::vector<NodeLocalStats> g_localStats;

static Ipv4Address g_broadcastAddress;

static uint32_t g_numNodes = 60;
static uint32_t g_source = 0;
static uint32_t g_destination = 59;
static uint32_t g_packetSizeBytes = 400;
static uint32_t g_packetCount = 200;
static double g_packetIntervalSec = 0.10;
static double g_commRange = 180.0;
static double g_speedMin = 2.0;
static double g_speedMax = 15.0;
static double g_minProgressMeters = -5.0;
static uint32_t g_candidateFanout = 4;
static uint32_t g_maxTtl = 20;
static double g_initialEnergyJ = 200.0;
static double g_lowEnergyThreshold = 0.25;
static double g_dropPenaltyWeight = 0.60;
static double g_energyPenaltyWeight = 0.40;

static uint32_t g_sent = 0;
static uint32_t g_delivered = 0;
static uint32_t g_forwarded = 0;
static uint32_t g_duplicateDrops = 0;
static uint32_t g_ttlDrops = 0;
static uint32_t g_noCandidateDrops = 0;
static uint32_t g_energyDrops = 0;
static uint32_t g_suppressedForwards = 0;
static double g_endTimeSec = 90.0;
static double g_delaySumSec = 0.0;
static uint32_t g_forwardJitterUs = 2500;
static bool g_enableAdaptivePacing = true;
static double g_adaptivePacingGain = 2.0;
static Ptr<UniformRandomVariable> g_jitterRv = CreateObject<UniformRandomVariable>();
static double g_epsilonBase = 0.08;
static double g_epsilonMobilityWeight = 0.35;
static double g_epsilonVarianceWeight = 0.30;
static double g_epsilonDropWeight = 0.35;
static double g_epsilonMin = 0.05;
static double g_epsilonMax = 0.50;
static double g_epsilonDecayFactor = 120.0;
static uint32_t g_rewardVarianceWindow = 128;

static std::map<uint64_t, Time> g_sendTime;

static uint64_t
PacketKey(uint32_t src, uint32_t packetId)
{
  return (static_cast<uint64_t>(src) << 32) | static_cast<uint64_t>(packetId);
}

static void
UpdateLocalReward(uint32_t nodeId, double reward)
{
  auto& state = g_localStats[nodeId];
  state.rewards.push_back(reward);
  while (state.rewards.size() > g_rewardVarianceWindow)
    {
      state.rewards.pop_front();
    }

  if (state.rewards.size() < 2)
    {
      state.rewardVariance = 0.0;
      return;
    }

  double mean = 0.0;
  for (double r : state.rewards)
    {
      mean += r;
    }
  mean /= static_cast<double>(state.rewards.size());

  double var = 0.0;
  for (double r : state.rewards)
    {
      const double d = r - mean;
      var += d * d;
    }
  state.rewardVariance = var / static_cast<double>(state.rewards.size());
}

static double
GetLocalRewardVarianceSignal(uint32_t nodeId)
{
  if (nodeId >= g_localStats.size())
    {
      return 0.0;
    }
  return std::clamp(g_localStats[nodeId].rewardVariance / 0.5, 0.0, 1.0);
}

static double
GetLocalDropRateSignal(uint32_t nodeId)
{
  if (nodeId >= g_localStats.size())
    {
      return 0.0;
    }
  const auto& state = g_localStats[nodeId];
  if (state.selected == 0)
    {
      return 0.0;
    }
  return std::clamp(static_cast<double>(state.dropped) / static_cast<double>(state.selected), 0.0, 1.0);
}

static double
GetEnergyRatio(uint32_t nodeId)
{
  if (nodeId >= g_energySources.size() || g_initialEnergyJ <= 0.0)
    {
      return 1.0;
    }
  const double remaining = g_energySources[nodeId]->GetRemainingEnergy();
  return std::clamp(remaining / g_initialEnergyJ, 0.0, 1.0);
}

static double
GetMobilitySignal(uint32_t nodeId, uint32_t destinationId)
{
  const auto nodeMob = g_nodes.Get(nodeId)->GetObject<MobilityModel>();
  const auto dstMob = g_nodes.Get(destinationId)->GetObject<MobilityModel>();
  const Vector vn = nodeMob->GetVelocity();
  const Vector vd = dstMob->GetVelocity();
  const double relV =
      std::sqrt((vn.x - vd.x) * (vn.x - vd.x) + (vn.y - vd.y) * (vn.y - vd.y) + (vn.z - vd.z) * (vn.z - vd.z));
  const double norm = std::max(1.0, g_speedMax);
  return std::clamp(relV / norm, 0.0, 1.0);
}

static double
GetLinkStability(uint32_t a, uint32_t b)
{
  const auto ma = g_nodes.Get(a)->GetObject<MobilityModel>();
  const auto mb = g_nodes.Get(b)->GetObject<MobilityModel>();
  const Vector va = ma->GetVelocity();
  const Vector vb = mb->GetVelocity();
  const double dot = (va.x * vb.x) + (va.y * vb.y) + (va.z * vb.z);
  const double na = std::sqrt((va.x * va.x) + (va.y * va.y) + (va.z * va.z));
  const double nb = std::sqrt((vb.x * vb.x) + (vb.y * vb.y) + (vb.z * vb.z));
  if (na < 1e-6 || nb < 1e-6)
    {
      return 0.5;
    }
  const double cosine = dot / (na * nb);
  return std::clamp(0.5 * (cosine + 1.0), 0.0, 1.0);
}

static std::vector<uint32_t>
GetCandidates(uint32_t currentNode, uint32_t destinationNode, uint32_t previousHop, uint64_t packetKey)
{
  std::vector<std::pair<double, uint32_t>> scored;

  const auto curMob = g_nodes.Get(currentNode)->GetObject<MobilityModel>();
  const auto dstMob = g_nodes.Get(destinationNode)->GetObject<MobilityModel>();
  const double myDistToDst = curMob->GetDistanceFrom(dstMob);

  for (uint32_t j = 0; j < g_numNodes; ++j)
    {
      if (j == currentNode || j == previousHop)
        {
          continue;
        }
      if (g_seenPackets[j].find(packetKey) != g_seenPackets[j].end())
        {
          continue; // Avoid selecting nodes that already saw this packet.
        }

      const auto nbrMob = g_nodes.Get(j)->GetObject<MobilityModel>();
      const double linkDistance = curMob->GetDistanceFrom(nbrMob);
      if (linkDistance > g_commRange)
        {
          continue;
        }

      const double nbrDistToDst = nbrMob->GetDistanceFrom(dstMob);
      const double progress = myDistToDst - nbrDistToDst;

      if (progress >= g_minProgressMeters)
        {
          const double progressNorm = std::clamp((progress + 5.0) / std::max(1.0, g_commRange + 5.0), 0.0, 1.0);
          const double qNorm = 0.5 + (0.5 * std::tanh(g_learner.GetQValue(currentNode, j)));
          const double energy = GetEnergyRatio(j);
          const double stability = GetLinkStability(currentNode, j);
          const double score = (0.45 * progressNorm) + (0.25 * qNorm) + (0.20 * energy) + (0.10 * stability);
          scored.push_back({score, j});
        }
    }

  if (scored.empty())
    {
      return {};
    }

  std::sort(scored.begin(),
            scored.end(),
            [](const std::pair<double, uint32_t>& a, const std::pair<double, uint32_t>& b) { return a.first > b.first; });

  std::vector<uint32_t> ranked;
  const size_t k = std::min<size_t>(g_candidateFanout, scored.size());
  for (size_t idx = 0; idx < k; ++idx)
    {
      ranked.push_back(scored[idx].second);
    }

  if (ranked.size() > 1)
    {
      g_learner.SetHybridSignals(GetMobilitySignal(currentNode, destinationNode),
                                 GetLocalRewardVarianceSignal(currentNode),
                                 GetLocalDropRateSignal(currentNode));
      const double epsilon = g_learner.ComputeEpsilon(Simulator::Now().GetSeconds());
      if (g_jitterRv->GetValue(0.0, 1.0) < epsilon)
        {
          const uint32_t idx = static_cast<uint32_t>(g_jitterRv->GetInteger(1, ranked.size() - 1));
          std::swap(ranked[0], ranked[idx]);
        }
    }
  return ranked;
}

static void
BroadcastPacket(uint32_t senderNode, const OpportunisticHeader& header)
{
  Ptr<Packet> p = Create<Packet>(g_packetSizeBytes);
  p->AddHeader(header);

  InetSocketAddress dstAddr(g_broadcastAddress, 9999);
  g_sockets[senderNode]->SendTo(p, 0, dstAddr);
}

static double
GetNodeDropRate(uint32_t nodeId)
{
  if (nodeId >= g_localStats.size() || g_localStats[nodeId].selected == 0)
    {
      return 0.0;
    }
  return static_cast<double>(g_localStats[nodeId].dropped) / static_cast<double>(g_localStats[nodeId].selected);
}

static double
GetLowEnergyPenalty(uint32_t nodeId)
{
  if (nodeId >= g_energySources.size() || g_initialEnergyJ <= 0.0)
    {
      return 0.0;
    }

  const double remaining = g_energySources[nodeId]->GetRemainingEnergy();
  const double ratio = std::max(0.0, std::min(1.0, remaining / g_initialEnergyJ));

  if (ratio >= g_lowEnergyThreshold || g_lowEnergyThreshold <= 0.0)
    {
      return 0.0;
    }

  return (g_lowEnergyThreshold - ratio) / g_lowEnergyThreshold;
}

static bool
HasEnergy(uint32_t nodeId)
{
  if (nodeId >= g_energySources.size())
    {
      return true;
    }
  return g_energySources[nodeId]->GetRemainingEnergy() > 1e-6;
}

static double
ComputeReward(uint32_t node, uint32_t chosenNext, uint32_t destination)
{
  const auto nodeMob = g_nodes.Get(node)->GetObject<MobilityModel>();
  const auto nextMob = g_nodes.Get(chosenNext)->GetObject<MobilityModel>();
  const auto dstMob = g_nodes.Get(destination)->GetObject<MobilityModel>();

  const double before = nodeMob->GetDistanceFrom(dstMob);
  const double after = nextMob->GetDistanceFrom(dstMob);
  const double progress = (before - after) / std::max(1.0, g_commRange);

  const double dropRiskPenalty = g_dropPenaltyWeight * GetNodeDropRate(chosenNext);
  const double lowEnergyPenalty = g_energyPenaltyWeight * GetLowEnergyPenalty(chosenNext);
  const double energyReward = 0.10 * GetEnergyRatio(chosenNext);
  const double stabilityReward = 0.10 * GetLinkStability(node, chosenNext);
  const double congestionPenalty =
      0.08 * std::min(1.0, static_cast<double>(g_pendingForwards[chosenNext].size()) / 8.0);

  double reward = progress + energyReward + stabilityReward - 0.02 - dropRiskPenalty - lowEnergyPenalty -
                  congestionPenalty; // progress + reliability terms - costs
  if (chosenNext == destination)
    {
      reward += 1.0;
    }
  return reward;
}

static void
ScheduleForwarding(uint32_t nodeId, uint64_t key, const OpportunisticHeader& rxHeader)
{
  const std::vector<uint32_t> candidates = rxHeader.GetCandidates();
  uint32_t rank = candidates.size();
  for (uint32_t i = 0; i < candidates.size(); ++i)
    {
      if (candidates[i] == nodeId)
        {
          rank = i;
          break;
        }
    }
  if (rank >= candidates.size())
    {
      return;
    }

  const uint32_t prevHop = rxHeader.GetPreviousHop();
  const auto prevMob = g_nodes.Get(prevHop)->GetObject<MobilityModel>();
  const auto nodeMob = g_nodes.Get(nodeId)->GetObject<MobilityModel>();
  const auto dstMob = g_nodes.Get(rxHeader.GetDst())->GetObject<MobilityModel>();

  const double progress = prevMob->GetDistanceFrom(dstMob) - nodeMob->GetDistanceFrom(dstMob);
  const double progressNorm = std::clamp((progress + 5.0) / std::max(1.0, g_commRange + 5.0), 0.0, 1.0);
  const double qNorm = 0.5 + (0.5 * std::tanh(g_learner.GetQValue(prevHop, nodeId)));
  const double energyNorm = GetEnergyRatio(nodeId);
  const double rankPenalty =
      (candidates.size() > 1) ? (static_cast<double>(rank) / static_cast<double>(candidates.size() - 1)) : 0.0;

  const double baseDelayMs = 1.0;
  const double scoreDelayMs =
      8.0 * (0.45 * (1.0 - qNorm) + 0.35 * (1.0 - progressNorm) + 0.20 * (1.0 - energyNorm) + 0.25 * rankPenalty);
  const double jitterMs = static_cast<double>(g_jitterRv->GetInteger(0, g_forwardJitterUs)) / 1000.0;
  const double totalDelayMs = baseDelayMs + scoreDelayMs + jitterMs;

  OpportunisticHeader header = rxHeader;
  EventId event = Simulator::Schedule(MilliSeconds(totalDelayMs), [nodeId, key, header]() mutable {
    auto itPending = g_pendingForwards[nodeId].find(key);
    if (itPending == g_pendingForwards[nodeId].end())
      {
        return; // Suppressed by overhearing another forwarder
      }
    g_pendingForwards[nodeId].erase(itPending);

    if (header.GetTtl() <= 0)
      {
        ++g_ttlDrops;
        ++g_localStats[nodeId].dropped;
        g_learner.UpdateQ(header.GetPreviousHop(), nodeId, -1.0, nodeId, {});
        return;
      }
    if (!HasEnergy(nodeId))
      {
        ++g_energyDrops;
        ++g_localStats[nodeId].dropped;
        g_learner.UpdateQ(header.GetPreviousHop(), nodeId, -1.2, nodeId, {});
        return;
      }

    std::vector<uint32_t> nextCandidates = GetCandidates(nodeId, header.GetDst(), header.GetPreviousHop(), key);
    if (nextCandidates.empty())
      {
        ++g_noCandidateDrops;
        ++g_localStats[nodeId].dropped;
        g_learner.UpdateQ(header.GetPreviousHop(), nodeId, -0.8, nodeId, {});
        return;
      }

    const double reward = ComputeReward(header.GetPreviousHop(), nodeId, header.GetDst());
    g_learner.UpdateQ(header.GetPreviousHop(), nodeId, reward, nodeId, nextCandidates);
    UpdateLocalReward(nodeId, reward);

    header.SetPreviousHop(nodeId);
    header.SetHopCount(header.GetHopCount() + 1);
    header.SetTtl(header.GetTtl() - 1);
    header.SetCandidates(nextCandidates);

    BroadcastPacket(nodeId, header);
    ++g_forwarded;
  });

  g_pendingForwards[nodeId][key] = event;
}

static void
ReceivePacket(uint32_t nodeId, Ptr<Socket> socket)
{
  Address from;
  Ptr<Packet> packet;

  while ((packet = socket->RecvFrom(from)))
    {
      OpportunisticHeader header;
      packet->RemoveHeader(header);

      const uint64_t key = PacketKey(header.GetSrc(), header.GetPacketId());
      auto pendingIt = g_pendingForwards[nodeId].find(key);
      if (pendingIt != g_pendingForwards[nodeId].end() && header.GetPreviousHop() != nodeId)
        {
          pendingIt->second.Cancel();
          g_pendingForwards[nodeId].erase(pendingIt);
          ++g_suppressedForwards;
        }

      if (!g_seenPackets[nodeId].insert(key).second)
        {
          if (nodeId == header.GetDst() || header.IsCandidate(nodeId) ||
              g_pendingForwards[nodeId].find(key) != g_pendingForwards[nodeId].end())
            {
              ++g_duplicateDrops;
            }
          continue;
        }

      if (nodeId == header.GetDst())
        {
          ++g_delivered;
          auto it = g_sendTime.find(key);
          if (it != g_sendTime.end())
            {
              g_delaySumSec += (Simulator::Now() - it->second).GetSeconds();
            }
          continue;
        }

      if (header.GetTtl() <= 0)
        {
          ++g_ttlDrops;
          continue;
        }
      if (!header.IsCandidate(nodeId))
        {
          continue; // Not in forwarding candidate list
        }

      ++g_localStats[nodeId].selected;
      if (!HasEnergy(nodeId))
        {
          ++g_energyDrops;
          ++g_localStats[nodeId].dropped;
          g_learner.UpdateQ(header.GetPreviousHop(), nodeId, -1.2, nodeId, {});
          continue;
        }

      ScheduleForwarding(nodeId, key, header);
    }
}

static void
GenerateTraffic(uint32_t packetId)
{
  const double dropSignal = GetLocalDropRateSignal(g_source);
  const double nextInterval =
      g_enableAdaptivePacing ? (g_packetIntervalSec * (1.0 + (g_adaptivePacingGain * dropSignal))) : g_packetIntervalSec;

  if (packetId >= g_packetCount)
    {
      return;
    }

  const uint64_t key = PacketKey(g_source, packetId + 1);
  std::vector<uint32_t> candidates = GetCandidates(g_source, g_destination, g_source, key);
  if (candidates.empty())
    {
      ++g_noCandidateDrops;
      Simulator::Schedule(Seconds(nextInterval), &GenerateTraffic, packetId + 1);
      return;
    }

  OpportunisticHeader header;
  header.SetBase(g_source, g_destination, packetId + 1, g_source, 0, static_cast<int32_t>(g_maxTtl));
  header.SetCandidates(candidates);

  BroadcastPacket(g_source, header);

  g_sendTime[key] = Simulator::Now();
  ++g_sent;

  Simulator::Schedule(Seconds(nextInterval), &GenerateTraffic, packetId + 1);
}

int
main(int argc, char* argv[])
{
  CommandLine cmd;
  cmd.AddValue("nodes", "Number of MANET nodes", g_numNodes);
  cmd.AddValue("source", "Source node ID", g_source);
  cmd.AddValue("destination", "Destination node ID", g_destination);
  cmd.AddValue("commRange", "Communication range for candidate discovery (meters)", g_commRange);
  cmd.AddValue("speedMin", "Minimum node speed for RandomWaypoint (m/s)", g_speedMin);
  cmd.AddValue("speedMax", "Maximum node speed for RandomWaypoint (m/s)", g_speedMax);
  cmd.AddValue("minProgressMeters", "Minimum progress toward destination for candidate relays (meters)", g_minProgressMeters);
  cmd.AddValue("candidateFanout", "Top-K relay candidates considered per hop", g_candidateFanout);
  cmd.AddValue("packets", "Number of packets generated by source", g_packetCount);
  cmd.AddValue("interval", "Packet generation interval (seconds)", g_packetIntervalSec);
  cmd.AddValue("simTime", "Total simulation time (seconds)", g_endTimeSec);
  cmd.AddValue("initialEnergyJ", "Initial energy per node (Joules)", g_initialEnergyJ);
  cmd.AddValue("lowEnergyThreshold", "Low-energy threshold ratio [0..1]", g_lowEnergyThreshold);
  cmd.AddValue("dropPenaltyWeight", "Reward penalty weight for drop-prone relays", g_dropPenaltyWeight);
  cmd.AddValue("energyPenaltyWeight", "Reward penalty weight for low-energy relays", g_energyPenaltyWeight);
  cmd.AddValue("epsilonBase", "Hybrid epsilon base value", g_epsilonBase);
  cmd.AddValue("epsilonMobilityWeight", "Hybrid epsilon mobility weight", g_epsilonMobilityWeight);
  cmd.AddValue("epsilonVarianceWeight", "Hybrid epsilon reward variance weight", g_epsilonVarianceWeight);
  cmd.AddValue("epsilonDropWeight", "Hybrid epsilon drop-rate weight", g_epsilonDropWeight);
  cmd.AddValue("epsilonMin", "Minimum epsilon clamp", g_epsilonMin);
  cmd.AddValue("epsilonMax", "Maximum epsilon clamp", g_epsilonMax);
  cmd.AddValue("epsilonDecayFactor", "Hybrid epsilon time-decay factor (seconds)", g_epsilonDecayFactor);
  cmd.AddValue("forwardJitterUs", "Max forwarding jitter in microseconds", g_forwardJitterUs);
  cmd.AddValue("enableAdaptivePacing", "Enable source adaptive pacing based on drop signal", g_enableAdaptivePacing);
  cmd.AddValue("adaptivePacingGain", "Adaptive pacing gain for drop signal", g_adaptivePacingGain);
  cmd.Parse(argc, argv);

  if (g_numNodes < 2)
    {
      NS_FATAL_ERROR("Need at least 2 nodes");
    }
  if (g_source >= g_numNodes || g_destination >= g_numNodes || g_source == g_destination)
    {
      NS_FATAL_ERROR("Invalid source/destination node IDs");
    }
  if (g_candidateFanout == 0)
    {
      NS_FATAL_ERROR("candidateFanout must be >= 1");
    }
  if (g_epsilonMax < g_epsilonMin)
    {
      NS_FATAL_ERROR("epsilonMax must be >= epsilonMin");
    }
  g_learner.ConfigureHybridEpsilon(g_epsilonBase,
                                   g_epsilonMobilityWeight,
                                   g_epsilonVarianceWeight,
                                   g_epsilonDropWeight,
                                   g_epsilonMin,
                                   g_epsilonMax,
                                   g_epsilonDecayFactor);

  g_nodes.Create(g_numNodes);

  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211g);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                               "DataMode",
                               StringValue("ErpOfdmRate24Mbps"),
                               "ControlMode",
                               StringValue("ErpOfdmRate6Mbps"));

  YansWifiChannelHelper channel;
  channel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  channel.AddPropagationLoss("ns3::RangePropagationLossModel",
                             "MaxRange",
                             DoubleValue(g_commRange));
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());

  WifiMacHelper mac;
  mac.SetType("ns3::AdhocWifiMac");

  NetDeviceContainer devices = wifi.Install(phy, mac, g_nodes);

  BasicEnergySourceHelper basicSourceHelper;
  basicSourceHelper.Set("BasicEnergySourceInitialEnergyJ", DoubleValue(g_initialEnergyJ));
  energy::EnergySourceContainer energyContainer = basicSourceHelper.Install(g_nodes);

  WifiRadioEnergyModelHelper radioEnergyHelper;
  radioEnergyHelper.Set("TxCurrentA", DoubleValue(0.28));
  radioEnergyHelper.Set("RxCurrentA", DoubleValue(0.22));
  radioEnergyHelper.Install(devices, energyContainer);

  g_energySources.clear();
  g_energySources.reserve(g_numNodes);
  for (uint32_t i = 0; i < g_numNodes; ++i)
    {
      g_energySources.push_back(energyContainer.Get(i));
    }

  MobilityHelper mobility;
  Ptr<RandomRectanglePositionAllocator> positionAlloc = CreateObject<RandomRectanglePositionAllocator>();
  positionAlloc->SetAttribute("X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=500.0]"));
  positionAlloc->SetAttribute("Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=500.0]"));
  mobility.SetPositionAllocator(positionAlloc);
  std::ostringstream speedExpr;
  speedExpr << "ns3::UniformRandomVariable[Min=" << g_speedMin << "|Max=" << g_speedMax << "]";

  // Random Waypoint is widely used for MANET dynamics with continuously changing neighbors.
  mobility.SetMobilityModel("ns3::RandomWaypointMobilityModel",
                            "Speed",
                            StringValue(speedExpr.str()),
                            "Pause",
                            StringValue("ns3::ConstantRandomVariable[Constant=0.25]"),
                            "PositionAllocator",
                            PointerValue(positionAlloc));
  mobility.Install(g_nodes);

  InternetStackHelper internet;
  internet.Install(g_nodes);

  Ipv4AddressHelper ipv4;
  ipv4.SetBase("10.10.0.0", "255.255.0.0");
  ipv4.Assign(devices);
  g_broadcastAddress = Ipv4Address("10.10.255.255");

  g_sockets.resize(g_numNodes);
  g_seenPackets.resize(g_numNodes);
  g_pendingForwards.assign(g_numNodes, {});
  g_localStats.assign(g_numNodes, {});

  for (uint32_t i = 0; i < g_numNodes; ++i)
    {
      g_sockets[i] = Socket::CreateSocket(g_nodes.Get(i), UdpSocketFactory::GetTypeId());
      g_sockets[i]->SetAllowBroadcast(true);
      g_sockets[i]->Bind(InetSocketAddress(Ipv4Address::GetAny(), 9999));
      g_sockets[i]->SetRecvCallback(MakeBoundCallback(&ReceivePacket, i));
    }

  Simulator::Schedule(Seconds(1.0), &GenerateTraffic, 0);

  Simulator::Stop(Seconds(g_endTimeSec));
  Simulator::Run();

  double avgRemainingEnergyJ = 0.0;
  for (uint32_t i = 0; i < g_numNodes && i < g_energySources.size(); ++i)
    {
      avgRemainingEnergyJ += g_energySources[i]->GetRemainingEnergy();
    }
  avgRemainingEnergyJ /= std::max(1u, g_numNodes);

  Simulator::Destroy();

  const double pdr = (g_sent > 0) ? static_cast<double>(g_delivered) / static_cast<double>(g_sent) : 0.0;
  const double avgDelay = (g_delivered > 0) ? (g_delaySumSec / static_cast<double>(g_delivered)) : 0.0;
  const double throughputMbps = (g_delivered * g_packetSizeBytes * 8.0) / (g_endTimeSec * 1e6);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "\n=== MANET Opportunistic Routing + Mean Field RL ===\n";
  std::cout << "Nodes: " << g_numNodes << ", Source: " << g_source << ", Destination: " << g_destination << "\n";
  std::cout << "Mobility speed range (m/s): [" << g_speedMin << ", " << g_speedMax
            << "], minProgressMeters: " << g_minProgressMeters << ", candidateFanout: " << g_candidateFanout
            << "\n";
  std::cout << "Forward jitter (us): " << g_forwardJitterUs << ", adaptive pacing: " << g_enableAdaptivePacing
            << ", pacing gain: " << g_adaptivePacingGain << "\n";
  std::cout << "Packets sent: " << g_sent << ", delivered: " << g_delivered << ", forwarded: " << g_forwarded << "\n";
  std::cout << "PDR: " << pdr << ", Avg delay (s): " << avgDelay << ", Throughput (Mbps): " << throughputMbps << "\n";
  std::cout << "Drops - duplicate: " << g_duplicateDrops << ", ttl: " << g_ttlDrops
            << ", no-candidate: " << g_noCandidateDrops << ", energy: " << g_energyDrops
            << ", suppressed-forwards: " << g_suppressedForwards << "\n";
  std::cout << "Penalty weights - drop: " << g_dropPenaltyWeight << ", energy: " << g_energyPenaltyWeight
            << ", avg remaining energy (J): " << avgRemainingEnergyJ << "\n";
  std::cout << "Hybrid epsilon = base + wm*mobility + wv*rewardVariance + wd*dropRate\n";
  std::cout << "Hybrid epsilon params: base=" << g_epsilonBase << ", wm=" << g_epsilonMobilityWeight
            << ", wv=" << g_epsilonVarianceWeight << ", wd=" << g_epsilonDropWeight
            << ", min=" << g_epsilonMin << ", max=" << g_epsilonMax << ", decay=" << g_epsilonDecayFactor << "\n";

  // Append per-run results for automated experiment sweeps.
  std::ofstream csvFile;
  csvFile.open("manet_results.csv", std::ios::app);
  if (csvFile.is_open())
    {
      std::ifstream checkFile("manet_results.csv", std::ios::binary | std::ios::ate);
      const bool writeHeader = !checkFile.good() || checkFile.tellg() == 0;
      if (writeHeader)
        {
          csvFile << "nodes,speed_mps,sent,delivered,forwarded,pdr,avg_delay_s,throughput_mbps,"
                     "duplicate_drops,ttl_drops,no_candidate_drops,energy_drops,suppressed_forwards\n";
        }

      csvFile << g_numNodes << "," << g_speedMax << "," << g_sent << "," << g_delivered << "," << g_forwarded << ","
              << pdr << "," << avgDelay << "," << throughputMbps << "," << g_duplicateDrops << "," << g_ttlDrops
              << "," << g_noCandidateDrops << "," << g_energyDrops << "," << g_suppressedForwards << "\n";
      csvFile.close();
    }

  return 0;
}
