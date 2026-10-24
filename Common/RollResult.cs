using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using YamlDotNet.Serialization;

namespace Common
{
    public readonly struct RollResult
    {
        private readonly byte _first;
        private readonly byte _second;

        public readonly byte First
        {
            get { return _first; }
            init
            {
                if (value >= 1 && value <= 6)
                {
                    _first = value;
                }
            }
        }

        public readonly byte Second
        {
            get { return _second; }
            init
            {
                if (value >= 1 && value <= 6)
                {
                    _second = value;
                }
            }
        }

        [YamlIgnore]
        public readonly byte Total
        {
            get { return (byte)(First + Second); }
        }

        public RollResult()
        {

        }

        public static RollResult GetRandom()
        {
            return new RollResult()
            {
                First = (byte)Utils.Random.NextDie(),
                Second = (byte)Utils.Random.NextDie()
            };
        }

        public override bool Equals(object? obj)
        {
            return obj is RollResult result &&
                   _first == result._first &&
                   _second == result._second;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(_first, _second);
        }

        public static bool operator ==(RollResult left, RollResult right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(RollResult left, RollResult right)
        {
            return !(left == right);
        }
    }
}
