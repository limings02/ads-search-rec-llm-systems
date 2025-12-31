"""Auction stream schema: defines bidding events and auction flow for replay simulation."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class AuctionType(str, Enum):
    """Auction type enumeration."""
    FIRST_PRICE = "first_price"
    SECOND_PRICE = "second_price"
    VCG = "vcg"


@dataclass
class BidEvent:
    """A single bid event in an auction."""
    auction_id: str
    bidder_id: str
    bid_price: float
    advertiser_id: str
    winning: Optional[bool] = None
    impression: Optional[bool] = None
    click: Optional[bool] = None
    conversion: Optional[bool] = None
    market_price: Optional[float] = None  # For second-price auction
    
    # Feature context
    features: Dict[str, Any] = field(default_factory=dict)
    
    # CTR/CVR predictions (model outputs)
    predicted_ctr: Optional[float] = None
    predicted_cvr: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AuctionRecord:
    """A single auction record with all participants."""
    auction_id: str
    timestamp: int
    num_slots: int = 1
    auction_type: AuctionType = AuctionType.SECOND_PRICE
    
    # All bids in this auction
    bids: List[BidEvent] = field(default_factory=list)
    
    # True winner info (ground truth from data)
    winning_bid_id: Optional[str] = None
    clearing_price: Optional[float] = None
    
    # User/context features
    user_features: Dict[str, Any] = field(default_factory=dict)
    context_features: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["auction_type"] = self.auction_type.value
        data["bids"] = [bid.to_dict() for bid in self.bids]
        return data


@dataclass
class AuctionStream:
    """
    Auction stream: a sequence of auction records that can be replayed for simulation.
    Used for closed-loop evaluation (stage 2: simulation).
    """
    name: str
    version: str
    description: Optional[str] = None
    
    # Auction records
    auctions: List[AuctionRecord] = field(default_factory=list)
    
    # Metadata
    total_auctions: Optional[int] = None
    total_bids: Optional[int] = None
    total_impressions: Optional[int] = None
    total_clicks: Optional[int] = None
    total_conversions: Optional[int] = None
    
    # Time range
    start_timestamp: Optional[int] = None
    end_timestamp: Optional[int] = None
    
    # Dataset info
    dataset_name: Optional[str] = None
    split: Optional[str] = None  # train, valid, test
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_auction(self, auction: AuctionRecord) -> None:
        """Add an auction record."""
        self.auctions.append(auction)
    
    def __len__(self) -> int:
        """Return number of auctions."""
        return len(self.auctions)
    
    def __getitem__(self, index: int) -> AuctionRecord:
        """Get auction by index."""
        return self.auctions[index]
    
    def to_jsonl(self) -> str:
        """Convert to JSONL (one auction per line)."""
        lines = []
        for auction in self.auctions:
            lines.append(json.dumps(auction.to_dict(), ensure_ascii=False))
        return "\n".join(lines)
    
    @classmethod
    def from_jsonl(cls, jsonl_str: str, name: str, version: str) -> "AuctionStream":
        """Create from JSONL string."""
        stream = cls(name=name, version=version)
        lines = jsonl_str.strip().split("\n")
        
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            
            bids = [
                BidEvent(
                    auction_id=bid["auction_id"],
                    bidder_id=bid["bidder_id"],
                    bid_price=bid["bid_price"],
                    advertiser_id=bid["advertiser_id"],
                    winning=bid.get("winning"),
                    impression=bid.get("impression"),
                    click=bid.get("click"),
                    conversion=bid.get("conversion"),
                    market_price=bid.get("market_price"),
                    features=bid.get("features", {}),
                    predicted_ctr=bid.get("predicted_ctr"),
                    predicted_cvr=bid.get("predicted_cvr"),
                )
                for bid in data.get("bids", [])
            ]
            
            auction = AuctionRecord(
                auction_id=data["auction_id"],
                timestamp=data["timestamp"],
                num_slots=data.get("num_slots", 1),
                auction_type=AuctionType(data.get("auction_type", "second_price")),
                bids=bids,
                winning_bid_id=data.get("winning_bid_id"),
                clearing_price=data.get("clearing_price"),
                user_features=data.get("user_features", {}),
                context_features=data.get("context_features", {}),
            )
            stream.add_auction(auction)
        
        return stream
